# ModelRoute — privacy-first LLM router

ModelRoute was previously named BrainRoute. Existing installs use the `brainroute` Python package and commands, along with their existing configuration and state paths.

ModelRoute picks the right model for each request. It classifies the prompt, applies admission policy, scores every eligible local and cloud model, executes the winner with a fallback chain, records what happened, and feeds that back into the next decision.

Local by default. Cloud only when a local model genuinely cannot do the job — and it tells you when that happens.

It ships a browser UI, a CLI, and an OpenAI-compatible gateway, with **no third-party Python dependencies**.

## Quick start

Requires Python 3.11+.

```bash
python3 -m unittest discover -s tests -t .   # hermetic: no providers, no network
python3 -m brainroute.cli eval               # routing evals
python3 -m brainroute.cli route "Design a distributed ingestion pipeline" --explain
python3 -m brainroute.server                 # then open http://127.0.0.1:8765
```

Nothing above calls a model provider. `-t .` matters on the test command: it lets the `tests` package redirect state to a temporary directory instead of writing telemetry into your real `data/`.

To actually execute a request, add `--execute`:

```bash
ollama serve && ollama pull qwen2.5:7b
python3 -m brainroute.cli ask "Write a short project tagline" --model local-qwen2-5 --execute
```

## Privacy postures

A posture is a named stance on where data may go, so the common decision is one setting rather than four correlated flags.

| Posture | Cloud | Per-request opt-in | Use when |
| --- | --- | --- | --- |
| `open` | Competes freely | n/a | Capability and cost matter more than data locality |
| `local_first` **(default)** | Available, must earn it | n/a | Local by default, without losing hard tasks |
| `local_only` | Rejected by policy | Allowed | Nothing leaves the machine unless you say so per request |
| `air_gapped` | Rejected, discovery disabled | **Refused** | No outbound traffic except your own local runtime |

```bash
python3 -m brainroute.cli posture              # show postures and the active one
python3 -m brainroute.cli posture local_only   # switch
python3 -m brainroute.cli policy               # resolved policy; * marks flags you pinned
```

Two rules hold under **every** posture, including `open`:

- A prompt classified as sensitive never goes to an external provider. `allow_cloud_for_private` is off in every preset.
- A per-request cloud opt-in lifts the blanket cloud block only. It never extends to sensitive prompts — opting in to cloud for a request is a different decision from consenting to send confidential data off the machine.

Under `local_only` a caller can escalate a single request explicitly, and the decision is flagged:

```bash
python3 -m brainroute.cli route "Design a distributed ingestion pipeline" --allow-cloud
```

`air_gapped` refuses that opt-in outright. A posture a single API call can switch off is not a guarantee.

### Postures are a starting point, not a cage

A posture supplies every policy flag you have *not* set yourself. Pin a flag explicitly and it survives posture switches — the UI marks pinned flags, `brainroute policy` marks them with `*`, and `brainroute posture --reset-overrides` hands control back.

This works because only your explicit values are written to `data/settings.json`:

```json
{
  "policy": { "posture": "local_only" },
  "routing": { "capability_floor": { "high": 0.7 } }
}
```

### When privacy is traded for capability, it says so

A local model that is cheap, fast, and private is still the wrong choice for work it cannot do. The **capability floor** enforces that. When the floor is what pushed a request off local hardware, the decision carries a warning rather than escalating silently:

```
Recommended: premium-reasoner (0.8214) [cloud]

! privacy_escalation: local-qwen2-5 would have won (0.9319 vs 0.8214) but fell
  below the capability floor for this task, so it routes to premium-reasoner.
  Lower routing.capability_floor.high to keep it local.
```

Every part of that is tunable: lower `routing.capability_floor.high` to keep hard work local at the cost of answer quality, raise `policy.local_bonus` to win more close races locally, or set `routing.warn_on_privacy_escalation` to `false` once you are happy with the trade.

The most effective fix is a bigger local model — see below.

## How a request is routed

```
prompt
  -> classify        task type, complexity, privacy, urgency, context size
  -> policy          hard constraints: context window, budget, privacy, circuit breakers
  -> score           task-modulated weights x (declared scores blended with telemetry)
  -> execute         selected model, then the ranked fallback chain
  -> record          latency, tokens, cost, success -> feeds the next decision
```

### 1. Classify

A keyword-and-phrase heuristic runs always and costs microseconds. Optionally a local model does structured classification against a JSON schema; results are cached, every field is validated against the heuristic, and any failure degrades to the heuristic rather than failing the request. Privacy and urgency signals the heuristic detected are never downgraded by the model — under-detecting either is the costlier error.

### 2. Policy (hard constraints)

Policy decides what is *eligible*, not what is *preferred*. A model is rejected if it cannot hold the prompt, would breach a budget, would leak a sensitive request, or its provider circuit is open. Every rejection is reported with a reason. The active posture supplies the cloud-access rules.

If *every* model is rejected for reliability reasons, the unhealthy ones are re-admitted — a degraded answer beats no answer.

### 3. Score (preferences)

Five factors — quality, speed, cost, privacy, task fit — each 0-1.

- **Weights are normalised**, so profiles can use intuitive numbers that need not sum to 1, and scores stay comparable across profiles.
- **Weights are then tilted by the task.** A high-complexity request shifts weight toward quality and task fit and away from cost and speed; an urgent one toward speed; a sensitive one toward privacy. Weights are renormalised after tilting.
- **Declared scores are blended with telemetry.** Speed moves from the configured prior toward measured p95 latency as samples accumulate. Quality is corrected by thumbs up/down feedback once enough votes exist.
- **Capability floors** stop a cheap, fast, private model from winning a task it cannot do, via a penalty proportional to the shortfall. Floors and penalty are configurable under `routing`.
- **The local bonus** (`policy.local_bonus`, default 0.15) applies when `prefer_local` is on. Deliberately sized to win close races without overpowering the capability floor.

`--explain` prints the full arithmetic:

```
$ brainroute route "Design a complex production architecture for a routing layer" --explain

Recommended: premium-reasoner (0.8214) [cloud]
Fallbacks:   gpt-5.4-mini -> local-qwen2-5
Task:        reasoning / high (confidence 0.74, via heuristic)
Margin:      +0.0869
Weights:     quality=0.46  speed=0.11  cost=0.09  privacy=0.11  task_fit=0.22

Score breakdown:

  premium-reasoner  final=0.8214  base=0.7614
    quality    0.950 x 0.465 = +0.4417
    speed      0.550 x 0.111 = +0.0609
    cost       0.345 x 0.092 = +0.0319
    privacy    0.350 x 0.111 = +0.0387
    task_fit   0.850 x 0.221 = +0.1882
    capability_headroom  +0.0600  comfortable headroom for a high-complexity task

  local-qwen2-5  final=0.7069  base=0.7819
    quality    0.650 x 0.465 = +0.3022
    ...
    capability_floor     -0.2250  below the quality floor (0.65 < 0.80) for a high-complexity task
    policy_bonus         +0.1500  policy prefers local providers
```

### 4. Execute

The selected model is tried first, then each model in the ranked fallback chain. Transient failures (429, 5xx, connection errors, timeouts) are retried with jittered exponential backoff; permanent ones (401, 400) are not. Streaming is never retried once bytes have reached the client — a silent retry would duplicate output.

### 5. Learn

Every attempt writes a run record: latency, token counts, estimated cost, success. Those records drive:

- **Speed and quality blending** — measured behaviour gradually overrides declared priors.
- **Circuit breakers** — after N consecutive failures a model is removed from routing, then half-opened after a cooldown to probe recovery.
- **Budget enforcement** — month-to-date spend is real, not estimated from a config file.

Optional epsilon-greedy **exploration** occasionally promotes a near-tied runner-up so the router keeps learning about models it would otherwise never pick. Off by default, so routing stays deterministic and evals reproducible.

`brainroute reliability` shows observed success rates, p95 latency, and circuit state.

## Models and providers

Three provider adapters cover everything: `ollama` for local, `openai` for the Responses API, and `openai_compatible` for anything else.

### Several local models side by side

Every model your local runtime reports becomes its own routing candidate with a distinct id, so `qwen2.5:7b`, `qwen2.5-coder:14b`, and `llama3.3:70b` compete with each other, not just with cloud. Capability is derived from parameter count, and models under 30B are marked weak for `legal` and `medical` work. Discovered models arrive disabled until you enable them.

This is the most effective way to keep work local: a 32B scores 0.824 and a 70B 0.85, both above the default 0.80 capability floor. Install one and hard tasks stop escalating.

```
id                          params  quality  speed     ctx  strengths
ollama-qwen2-5-7b              7.6    0.613  0.745    32k   coding,general,private,summarization
ollama-qwen2-5-coder-14b      14.8    0.709  0.678    32k   coding,general,private,summarization
ollama-qwen2-5-32b            32.8    0.824  0.598    32k   coding,general,private,summarization
ollama-llama3-3-70b           70.6    0.850  0.520   128k   general,private,summarization
```

**Context windows are detected, not guessed.** The tags endpoint does not report context length, so discovery asks the runtime for each model's real window (in parallel, best-effort, cached), falling back to a table of known families. This matters because an under-reported window escalates long prompts to cloud that a local model could have handled. The UI marks any window that is still an estimate.

**Requests ask for the window they were promised.** Ollama defaults `num_ctx` to about 4k regardless of what a model supports, and silently truncates beyond it — so correct detection alone would trade a visible escalation for an invisible quality loss. The adapter sizes `num_ctx` to what each request needs, in buckets, capped at the model's window:

| Prompt | Eligible models | `num_ctx` requested |
| --- | --- | --- |
| 2k | 32B, 70B, cloud | default (no option sent) |
| 9k | 32B, 70B, cloud | 16384 |
| 20k | 32B, 70B, cloud | 32768 |
| 60k | 70B, cloud | 65536 |

Memory therefore scales with the prompt rather than the model's ceiling — a 5k prompt on a 128k model does not allocate a 128k KV cache. Set `catalog.detect_local_context` to `false` to skip the probe and rely on the family table alone.

### Any OpenAI-compatible endpoint

The `openai_compatible` provider reaches anything speaking `/v1/chat/completions` — self-hosted vLLM or llama.cpp, LM Studio, a managed inference gateway, a corporate proxy, or another ModelRoute. Adding one is a config change, not new code:

```yaml
  - id: my-frontier-model
    provider: openai_compatible
    provider_label: my-gateway
    model: vendor/some-model
    endpoint: https://gateway.internal/v1/chat/completions
    api_key_env: MY_GATEWAY_KEY
    local: false
    enabled: true
    quality_score: 0.93
    speed_score: 0.70
    privacy_score: 0.35
    input_price_per_mtok: 3.00
    output_price_per_mtok: 15.00
    max_context: 200000
```

Use `requires_api_key: false` for endpoints with no credential, `headers` for static headers, and `extra_body` for endpoint-specific sampling knobs. Streaming, retries, fallback, health checks, and cost tracking work the same as for any other provider.

Set `max_context` — without it the context-window admission check cannot protect that model from an oversized prompt.

The seeded `premium-reasoner` is a placeholder of exactly this shape. It routes as a high-capability cloud option but cannot execute until you give it an `endpoint`, at which point it fails fast with a clear message rather than pretending to work.

### Remote catalog sources

None are configured by default. Any endpoint serving an OpenAI-style `/v1/models` list can be added under `catalog.remote_sources`; its models arrive as **disabled** candidates and nothing routes to them until you enable one.

```json
{
  "catalog": {
    "remote_sources": [
      {
        "id": "my-gateway",
        "enabled": true,
        "models_url": "https://gateway.internal/v1/models",
        "chat_url": "https://gateway.internal/v1/chat/completions",
        "api_key_env": "MY_GATEWAY_KEY",
        "limit": 25
      }
    ]
  }
}
```

Set `limit` — a large catalog will otherwise import hundreds of models, inflating `/api/config` and the model picker. Discovered models take pricing and context length from the source when reported; where prices are reported, `quality_score` is **inferred from price**, a crude proxy worth reviewing before enabling anything for serious work.

Discovery is outbound traffic even though it carries no prompt data, so `air_gapped` blocks it as a hard constraint — even if a source is pinned enabled.

## Configuration

| File | Purpose |
| --- | --- |
| [config/models.yaml](config/models.yaml) | Model registry: capabilities, priors, prices, context windows |
| [config/router.weights.yaml](config/router.weights.yaml) | Routing profiles |
| [config/evals.yaml](config/evals.yaml) | Routing evals |
| `data/settings.json` | Only the settings you explicitly changed |

Profiles make trade-offs explicit:

- `balanced` — quality, speed, cost, and privacy all considered
- `cheap` — prefers low-cost models
- `urgent` — prefers low-latency models
- `private` — heavily prefers local models

### Cost model

`cost_score` is **derived from prices**, not hand-tuned. Set `input_price_per_mtok` and `output_price_per_mtok` (USD per 1M tokens); the router computes a cost score on a logarithmic curve, estimates per-request spend before committing, and enforces per-request and monthly budgets against real recorded spend. Prices for well-known models come from a built-in table; discovered models carry their own when the source reports it.

A model with no price signal scores mid-market (0.5) rather than free, so an unpriced model never wins the cheap profile by accident.

### Environment variables

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI credential |
| per-model `api_key_env` | Credential for any `openai_compatible` endpoint |
| `BRAINROUTE_API_KEY` | Require `Authorization: Bearer ...` on the gateway and state-changing admin endpoints |
| `BRAINROUTE_DATA_DIR` | Move SQLite state and settings off the checkout |
| `BRAINROUTE_CONFIG_DIR` | Move YAML config off the checkout |

## Web UI

At [http://127.0.0.1:8765](http://127.0.0.1:8765): enter a prompt, choose a profile, inspect the ranked decision and why each model scored as it did, execute or stream into a persisted conversation, compare two candidates side by side, discover and enable local models, switch privacy posture, tune capability floors and budgets, watch provider health and circuit state, run evals, and record good/bad feedback that feeds back into routing.

## Gateway API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/healthz` | Liveness |
| GET | `/api/config` | Profiles, models, postures, settings |
| GET | `/api/health` | Provider reachability (checked in parallel) |
| GET | `/api/reliability` | Observed per-model behaviour and circuit state |
| GET | `/api/catalog` | Discovered models |
| GET | `/api/dashboard` | Aggregate run metrics |
| GET | `/api/evals` | Run routing evals |
| GET | `/api/telemetry`, `/api/sessions` | Event and conversation history |
| GET / POST | `/api/postures` | Read or switch the privacy posture |
| POST | `/api/route` | Route without executing |
| POST | `/api/ask` | Route and optionally execute |
| POST | `/api/chat/stream` | Server-sent event stream |
| POST | `/api/compare` | Run two candidates side by side |
| POST | `/api/feedback` | Record a good/bad rating |
| POST | `/api/models`, `/api/settings` | Enable a model, change settings |
| POST | `/api/maintenance/prune` | Trim telemetry to its retention cap |
| GET | `/v1/models` | OpenAI-compatible model list |
| POST | `/v1/chat/completions` | OpenAI-compatible completion, streaming supported |

`/api/route` and `/api/ask` accept either `prompt` or a full `messages` array, plus `"allow_cloud": true` to opt in to cloud for that request where the posture permits. Decisions return `local`, `policy.posture`, and a `warnings` array.

Point any OpenAI client at it:

```bash
curl http://127.0.0.1:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"brainroute-auto","messages":[{"role":"user","content":"Write a release note."}],"stream":true}'
```

Use `"model": "brainroute-auto"` to let the router choose, or name a model to pin it. Add `"brainroute_profile": "cheap"` to select a profile, or `"brainroute_allow_cloud": true` for a per-request cloud opt-in. Non-streaming responses include the full routing decision under a `brainroute` key.

## Evaluation

[config/evals.yaml](config/evals.yaml) holds routing evals that never call a provider. Beyond asserting which model wins, a case can pin a **posture**, assert the classification itself, require a specific warning, or require a minimum score margin — which catches the common failure where the router picks the right model for the wrong reason.

```yaml
- name: local_first_escalates_hard_work_and_warns
  prompt: Design a complex production architecture for a multi-provider AI routing layer.
  posture: local_first
  profile: balanced
  expected_model: premium-reasoner
  expected_warning: privacy_escalation
```

```bash
python3 -m brainroute.cli eval
```

Run it after any change to weights, model scores, prices, or postures.

## Production notes

- **Binding.** Run behind a host firewall or reverse proxy when binding beyond loopback. The server warns if you do so without an API key set.
- **Auth.** `BRAINROUTE_API_KEY` protects the gateway and every state-changing admin endpoint — including posture changes, the most security-relevant write the API exposes — compared in constant time. Read-only endpoints stay open for the local UI; set `gateway.protect_admin_api` to `false` to opt out.
- **Rate limiting.** Per-client fixed window over the API surface, configurable via `gateway.rate_limit_per_minute`. Static assets are exempt.
- **State.** SQLite in WAL mode with indexes and per-thread connections, falling back to a rollback journal on filesystems that cannot support WAL. `brainroute prune` caps table growth; the JSONL development log rotates at 16 MiB.
- **Caching.** Provider discovery, assembled config, and classification are TTL-cached, so a routing decision does not depend on live network calls. Settings writes invalidate the caches immediately.
- **Failure handling.** Every HTTP handler runs inside an error boundary that returns JSON rather than dropping the connection or leaking a traceback.
- **Credentials** belong in the process environment, never in config files.

## Maintenance (2026-11-06)

The bundled Qwen3 8B seed is available from https://ollama.com/library/qwen3:8b. GPT-5.4 mini pricing was checked on 2026-11-06 against https://developers.openai.com/api/docs/models/gpt-5.4-mini: $0.75 input and $4.50 output per million tokens, 400,000 context. Prices are estimates; review before budgeting paid workloads. Existing local model IDs stay stable for saved preferences.

The manually dispatched Publish package workflow uses PyPI trusted publishing. Configure the `pypi` GitHub environment and register this repository/workflow as a trusted publisher in the PyPI project before dispatching. CI builds wheel and source distributions without publishing.

Installed wheels include the seed model catalog and eval suite. Package installs store state in `~/.local/share/brainroute`; source checkouts retain `data/`. Override with `BRAINROUTE_DATA_DIR` and `BRAINROUTE_CONFIG_DIR`.
