# BrainPicker

BrainPicker is an early MVP for an intelligent model router: it looks at a request, scores available local and cloud models against your routing profile, and chooses the best model for the job.

The first version is intentionally small:

- editable YAML model registry
- weighted routing profiles for balanced, cheap, urgent, and private modes
- heuristic task classification
- deterministic model scoring
- dry-run CLI for seeing routing decisions
- local browser UI for testing prompts and profiles
- opt-in execution through local Ollama or OpenAI Responses API
- JSONL telemetry for route decisions

## Quick Start

```bash
python3 -m brainpicker.cli route "Refactor this Python module for readability"
python3 -m brainpicker.cli route "Summarize this private medical note" --profile private
python3 -m brainpicker.cli ask "Say hello in one sentence" --profile cheap --dry-run
python3 -m brainpicker.cli models
python3 -m brainpicker.cli health
python3 -m brainpicker.cli eval
```

Start the local test app:

```bash
python3 -m brainpicker.server
```

Then open [http://127.0.0.1:8765](http://127.0.0.1:8765).

Run tests:

```bash
python3 -m unittest discover -s tests
```

## Configuration

Model metadata lives in [config/models.yaml](/Users/gleis/Documents/BrainPicker/config/models.yaml).

Routing weights live in [config/router.weights.yaml](/Users/gleis/Documents/BrainPicker/config/router.weights.yaml).

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
ollama pull qwen3:8b
python3 -m brainpicker.cli ask "Write a short project tagline" --model local-qwen3 --execute
```

OpenAI:

```bash
export OPENAI_API_KEY="..."
python3 -m brainpicker.cli ask "Write a short project tagline" --model gpt-5.4-mini --execute
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

## Evaluation

BrainPicker includes a tiny routing eval set in [config/evals.yaml](/Users/gleis/Documents/BrainPicker/config/evals.yaml). Run it after changing model scores or routing weights:

```bash
python3 -m brainpicker.cli eval
```

This does not call any model providers. It checks whether the router chooses the expected model for representative prompts.

## Where This Goes Next

The useful product is not just a model picker. It becomes valuable when it learns from real usage:

- latency and error tracking per provider
- estimated cost per request
- user feedback on response quality
- fallback rules when a model fails
- a small classifier model for richer task detection
- a local dashboard for model performance and spend
