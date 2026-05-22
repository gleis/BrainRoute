# BrainRoute - LLM Router

BrainRoute is a local-first intelligent model router: it classifies a request, applies routing policy, scores enabled local and cloud models, executes the selected provider, records telemetry, and exposes a browser UI plus gateway APIs.

The current build includes:

- editable YAML seed registry plus runtime Ollama and optional OpenRouter discovery
- weighted routing profiles for balanced, cheap, urgent, and private modes
- heuristic classification with opt-in structured local router-model classification
- deterministic scoring, privacy filtering, local preference, and budget controls
- dry-run and execution CLI commands
- local browser UI for chat streaming, provider health, model enablement, compare, evals, policy, and feedback
- execution through Ollama, OpenAI Responses API, and enabled OpenRouter models
- SQLite telemetry, conversation storage, run metrics, and a JSONL development tail
- OpenAI-compatible gateway endpoints

## Quick Start

```bash
python3 -m brainroute.cli route "Refactor this Python module for readability"
python3 -m brainroute.cli route "Summarize this private medical note" --profile private
python3 -m brainroute.cli ask "Say hello in one sentence" --profile cheap --dry-run
python3 -m brainroute.cli models
python3 -m brainroute.cli health
python3 -m brainroute.cli eval
```

Start the local test app:

```bash
python3 -m brainroute.server
```

Then open [http://127.0.0.1:8765](http://127.0.0.1:8765).

Run tests:

```bash
python3 -m unittest discover -s tests
```

## Configuration

Model metadata lives in [config/models.yaml](config/models.yaml).

Routing weights live in [config/router.weights.yaml](config/router.weights.yaml).

Profiles let you make tradeoffs explicit:

- `balanced`: quality, speed, cost, and privacy are all considered
- `cheap`: prefers low-cost models
- `urgent`: prefers low-latency models
- `private`: heavily prefers local/private models

## Execution

Dry-run routing never calls a model. To actually call a provider, use `--execute`.

Ollama:

```bash
ollama serve
ollama pull qwen2.5:7b
python3 -m brainroute.cli ask "Write a short project tagline" --model local-qwen2-5 --execute
```

OpenAI:

```bash
export OPENAI_API_KEY="..."
python3 -m brainroute.cli ask "Write a short project tagline" --model gpt-5.4-mini --execute
```

The OpenAI adapter uses the Responses API (`POST /v1/responses`), which is the current primary text generation interface in OpenAI's API reference as of this scaffold.

## Web UI

The local UI lets you:

- enter a prompt
- choose a routing profile
- see the ranked model decision
- optionally execute the selected model
- inspect recent telemetry events
- check whether configured providers look available
- record quick good/bad feedback on a routing decision
- stream local provider output into a persisted conversation
- discover and enable installed Ollama models
- optionally discover OpenRouter models after enabling its catalog source
- choose a local structured classifier model
- run evals and compare top candidate routes
- set privacy and budget routing policy

Runtime settings are written to `data/settings.json` and SQLite state is stored in `data/brainroute.sqlite`. The `data/` directory is intentionally git-ignored.

## Gateway API

BrainRoute serves:

- `GET /healthz`
- `GET /api/config`, `/api/health`, `/api/catalog`, `/api/evals`, `/api/dashboard`
- `POST /api/route`, `/api/ask`, `/api/chat/stream`, `/api/compare`
- `GET /v1/models`
- `POST /v1/chat/completions`

Set `BRAINROUTE_API_KEY` to require `Authorization: Bearer ...` on the `/v1/*` gateway endpoints.

Example:

```bash
curl http://127.0.0.1:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"brainroute-auto","messages":[{"role":"user","content":"Write a release note."}]}'
```

For cloud execution set provider keys in the environment:

```bash
export OPENAI_API_KEY="..."
export OPENROUTER_API_KEY="..."
```

## Evaluation

BrainRoute includes a tiny routing eval set in [config/evals.yaml](config/evals.yaml). Run it after changing model scores or routing weights:

```bash
python3 -m brainroute.cli eval
```

This does not call any model providers. It checks whether the router chooses the expected model for representative prompts.

## Production Notes

BrainRoute has no third-party Python runtime dependency today. Run it behind a host firewall or a reverse proxy when binding beyond loopback, use `BRAINROUTE_API_KEY` for client access, set explicit routing policy before enabling external models, and keep provider credentials in the process environment rather than config files.
