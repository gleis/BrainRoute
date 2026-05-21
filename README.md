# BrainPicker

BrainPicker is an early MVP for an intelligent model router: it looks at a request, scores available local and cloud models against your routing profile, and chooses the best model for the job.

The first version is intentionally small:

- editable YAML model registry
- weighted routing profiles for balanced, cheap, urgent, and private modes
- heuristic task classification
- deterministic model scoring
- dry-run CLI for seeing routing decisions
- opt-in execution through local Ollama or OpenAI Responses API
- JSONL telemetry for route decisions

## Quick Start

```bash
python3 -m brainpicker.cli route "Refactor this Python module for readability"
python3 -m brainpicker.cli route "Summarize this private medical note" --profile private
python3 -m brainpicker.cli ask "Say hello in one sentence" --profile cheap --dry-run
```

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

## Where This Goes Next

The useful product is not just a model picker. It becomes valuable when it learns from real usage:

- latency and error tracking per provider
- estimated cost per request
- user feedback on response quality
- fallback rules when a model fails
- a small classifier model for richer task detection
- a local dashboard for model performance and spend
