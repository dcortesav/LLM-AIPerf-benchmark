# AI Performance Benchmarks

Benchmarking of large language model (LLM) inference performance using
[NVIDIA AIPerf](https://github.com/ai-dynamo/aiperf) against the Huawei MaaS
OpenAI-compatible endpoint.

## What is measured

Eight model/reasoning configurations are benchmarked under an identical
workload so the results are directly comparable:

| Model | Variants |
|-------|----------|
| GLM 5.2 | reasoning on / off |
| GLM 5.3 | reasoning on / low reasoning |
| DeepSeek v4.1-Flash | reasoning on / off |
| DeepSeek v4.1-Flash-1017 | reasoning on / off |

Metrics captured per configuration:

- **Time to First Token (TTFT)** — p50, p95, p99
- **Request latency** — p50, p95, p99
- **Input / output token consumption** — averages

## How it works

1. **`datasets/prompts.jsonl`** — 25 verbatim single-turn prompts (everyday
   tasks and coding tasks). Prompts are dispatched sequentially and wrap around,
   so every model receives the same prompt multiset in the same order.

2. **`benchmark.yaml`** — AIPerf configuration defining the shared workload
   shape: streaming enabled, builtin tokenizer, server-side token counts,
   3 warmup requests, 100 evaluation requests at a closed-loop concurrency of 5.

3. **`run_benchmarks.sh`** — Iterates over the eight configurations, invoking
   `aiperf profile` for each with per-model overrides (model id, reasoning
   payload, output directory). Results are written to a timestamped folder
   under `results/`.

4. **`generate_plots.py`** — Reads the AIPerf profiling-phase JSON summaries
   and produces comparison plots (grouped bar charts for TTFT and latency
   percentiles, horizontal bar charts for token consumption) exported as PNGs
   into `results/`.

## Usage

### Prerequisites

- Python 3.10+ virtual environment with dependencies installed:
  ```bash
  python -m venv .venv
  .venv/bin/pip install -r requirements.txt
  ```
- MaaS credentials exported in `~/.bashrc`:
  ```bash
  export MAAS_URL="https://<maas-endpoint>/v1/chat/completions"
  export MAAS_API_KEY="<your-api-key>"
  ```

### Running the benchmarks

```bash
./run_benchmarks.sh          # runs all 8 configurations
.venv/bin/python generate_plots.py   # generates comparison plots from the latest run
```

## Project structure

```
.
├── benchmark.yaml          # AIPerf workload configuration
├── run_benchmarks.sh       # sweep over all model/reasoning variants
├── generate_plots.py       # parse results and export comparison plots
├── datasets/
│   └── prompts.jsonl       # 25 verbatim benchmark prompts
├── requirements.txt        # Python dependencies
└── results/                # benchmark output (plots + raw telemetry)

    ├── Duration.png
    ├── Input_Tokens.png
    ├── Output_Tokens.png
    ├── Total_Token_Consumption.png
    └── TTFT.png
```
