# Ollama and vLLM Scenario Pipelines

These pipelines run MARBLE scenarios against one or more local models from a single command, using either Ollama or vLLM.

## What it does

- Loads scenario config patterns from `scripts/pipeline/ollama_pipeline.yaml`.
- Creates temporary per-run YAML configs under `logs/pipeline_tmp/`.
- Rewrites model fields (`llm`, `model`, `model_name`, `evaluate_llm`) for each selected model.
- Runs `marble/main.py --config_path <temp config>` for each task.
- Stores run logs and scenario outputs under `logs/pipeline/`.

## Quick start

From repo root:

### vLLM pipeline (new)

No-brainer GPU run:
```bash
python scripts/pipeline/run_vllm_pipeline.py --manage-vllm-server --model-set tiny
```

Start vLLM server on a GPU node (example):
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000
```

Start vLLM server on a CPU node (example):
```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --device cpu --host 0.0.0.0 --port 8000
```

Dry-run to preview tasks:
```bash
python scripts/pipeline/run_vllm_pipeline.py --dry-run
```

Run all scenarios with a model set:
```bash
python scripts/pipeline/run_vllm_pipeline.py --model-set tiny
python scripts/pipeline/run_vllm_pipeline.py --model-set small
python scripts/pipeline/run_vllm_pipeline.py --model-set medium
python scripts/pipeline/run_vllm_pipeline.py --model-set large
```

If vLLM is running on another host, override endpoint:
```bash
python scripts/pipeline/run_vllm_pipeline.py --model-set tiny --api-base http://<node-ip>:8000/v1
```

Evaluate the latest completed vLLM pipeline run:
```bash
python scripts/pipeline/run_vllm_evaluation_pipeline.py
```

Load large Hugging Face models one-by-one with automatic unload/VRAM cleanup:
```bash
python scripts/pipeline/run_vllm_pipeline.py \
  --manage-vllm-server \
  --model-set medium \
  --vllm-device cuda \
  --vllm-extra-args "--gpu-memory-utilization 0.92 --max-model-len 8192"
```

Set 2-GPU tensor parallel manually when needed:
```bash
python scripts/pipeline/run_vllm_pipeline.py \
  --manage-vllm-server \
  --model-set large \
  --vllm-device cuda \
  --vllm-extra-args "--tensor-parallel-size 2"
```

Notes:
- Add GPU selection manually when needed, for example: `--vllm-extra-args "--tensor-parallel-size 2"` together with environment variable `CUDA_VISIBLE_DEVICES=0,1`.

CPU-only node with managed vLLM server:
```bash
python scripts/pipeline/run_vllm_pipeline.py --manage-vllm-server --model-set tiny --vllm-device cpu
```

Low-disk mode (delete each model cache after it finishes):
```bash
python scripts/pipeline/run_vllm_pipeline.py --manage-vllm-server --model-set medium --vllm-device cuda --clear-hf-cache-after-model
```

Optional custom Hugging Face cache location:
```bash
python scripts/pipeline/run_vllm_pipeline.py --manage-vllm-server --model-set medium --vllm-device cuda --hf-cache-dir D:/hf_cache --clear-hf-cache-after-model
```

Gated/private Hugging Face model access:
```bash
python scripts/pipeline/run_vllm_pipeline.py --manage-vllm-server --models openai/meta-llama/Llama-3.1-8B-Instruct --hf-token <your_hf_token>
```

Notes for managed mode:
- Models in `scripts/pipeline/vllm_pipeline.yaml` can stay in LiteLLM-style form (`openai/<hf_repo_id>`).
- The runner strips `openai/` when launching `vllm serve` so Hugging Face repo IDs load correctly.
- Configure per-model parser/tool-calling flags in `vllm_model_overrides` inside `scripts/pipeline/vllm_pipeline.yaml`.
- Per-model `extra_args` are merged after CLI `--vllm-extra-args` and applied automatically when each model is served.
- After each model finishes, the runner terminates vLLM and performs best-effort VRAM cache cleanup.
- With `--clear-hf-cache-after-model`, the runner also deletes that model's local Hugging Face cache directory to free disk before the next model.

### Ollama pipeline

**Dry-run to preview tasks:**
```bash
python scripts/pipeline/run_ollama_pipeline.py --dry-run
```

**Run all scenarios with a model set:**
```bash
python scripts/pipeline/run_ollama_pipeline.py --model-set tiny
python scripts/pipeline/run_ollama_pipeline.py --model-set small
python scripts/pipeline/run_ollama_pipeline.py --model-set medium
```

**Evaluate the latest completed pipeline run:**
```bash
python scripts/pipeline/run_ollama_evaluation_pipeline.py
```

## Common usage

Run only selected scenarios:
```bash
python scripts/pipeline/run_ollama_pipeline.py --model-set tiny --scenarios reasoning,coding
```

Run across all 4 orchestration structures (star, graph, chain, tree):
```bash
python scripts/pipeline/run_ollama_pipeline.py --model-set tiny --orchestration all
```

Run only each config's default orchestration structure (no override):
```bash
python scripts/pipeline/run_ollama_pipeline.py --model-set tiny --orchestration default
```

Override models from CLI (takes precedence over --model-set):
```bash
python scripts/pipeline/run_ollama_pipeline.py --models openai/qwen2.5:0.5b,openai/llama3.1:8b
```

List available model sets:
```bash
grep -A 10 "^model_sets:" scripts/pipeline/ollama_pipeline.yaml
```

Inject additional recursive key overrides in generated configs:
```bash
python scripts/pipeline/run_ollama_pipeline.py --model-set small --set max_iterations=3
```

Stop on first failure:
```bash
python scripts/pipeline/run_ollama_pipeline.py --model-set tiny --fail-fast
```

Evaluate a specific run timestamp:
```bash
python scripts/pipeline/run_ollama_evaluation_pipeline.py --run 20260320_120737
```

Fail evaluation when successful tasks are missing outputs:
```bash
python scripts/pipeline/run_ollama_evaluation_pipeline.py --run 20260320_120737 --fail-on-missing-output
```

## Two-stage workflow

1. Run scenarios/models with `run_ollama_pipeline.py`.
2. Use `run_ollama_evaluation_pipeline.py` on that run directory.

`run_ollama_pipeline.py` now writes a run manifest file at:

`logs/pipeline/<timestamp>/tasks_manifest.json`

The evaluation pipeline reads that manifest and produces:

`logs/pipeline/<timestamp>/evaluation_summary.json`

The summary includes:
- Per-task run status (ok/failed/skipped)
- Output presence checks
- Existing evaluator payloads from each run output (`planning_scores`, `communication_scores`, `task_evaluation`, `code_quality`, etc.)
- Aggregate totals across all tasks

## Existing repo evaluators in stage 2

The evaluation pipeline does not invent new task metrics. It uses existing repository evaluations:

- For most scenarios, it reads evaluator outputs already produced during stage 1 runs.
- For `database`, it runs the existing `scripts/database/batch_eval.py` over reconstructed per-task JSON inputs.

Skip DB batch evaluator invocation if needed:
```bash
python scripts/pipeline/run_ollama_evaluation_pipeline.py --skip-db-batch-eval
```

## Compare multiple runs (CSV)

After stage 2 has produced `evaluation_summary.json` files, export comparison tables:

```bash
python scripts/pipeline/export_evaluation_comparison_csv.py
```

This writes two files under `logs/pipeline/comparison/`:
- `runs_<timestamp>.csv` (run-level comparison)
- `tasks_<timestamp>.csv` (task-level rows with existing evaluator payloads)

Compare specific runs only:

```bash
python scripts/pipeline/export_evaluation_comparison_csv.py --runs 20260320_120737,20260320_142148
```

For vLLM pipeline runs, use:

```bash
python scripts/pipeline/export_vllm_evaluation_comparison_csv.py
```

This writes CSV files under `logs/pipeline_vllm/comparison/`.

## Manifest structure

Edit `scripts/pipeline/ollama_pipeline.yaml`:

For vLLM runs, edit `scripts/pipeline/vllm_pipeline.yaml`.

### Global section
- `global.main_script`: Entry point, default `marble/main.py`.
- `global.env`: Environment variables applied to every run.
- `global.model_keys`: Keys to rewrite with target model.
- `global.orchestration_key`: Config key to rewrite when sweeping orchestration modes (default `coordinate_mode`).
- `global.orchestration_modes`: Modes used when `--orchestration all` (default list is `star`, `graph`, `chain`, `tree`).

### Model sets section
- `model_sets.<name>.description`: Human-readable description.
- `model_sets.<name>.models`: List of model IDs to run through all scenarios.

### Scenarios section
- `scenarios.<name>.configs`: Glob list for scenario configs.
- `scenarios.<name>.models`: Optional fallback models if no model set or --models is provided.
- `scenarios.<name>.max_configs`: Optional cap for matched configs.
- `scenarios.<name>.model_keys`: Optional scenario-specific rewrite keys.
- `scenarios.<name>.orchestration_key`: Optional scenario-specific orchestration key override.
- `scenarios.<name>.orchestration_modes`: Optional scenario-specific orchestration mode list.

## Precedence (which models get used)

1. `--models` from CLI (highest priority)
2. `--model-set` from CLI
3. `models:` defined in each scenario
4. Error if none of above are available

## Orchestration mode toggle

- `--orchestration default`: Runs with orchestration mode already in each source config.
- `--orchestration all`: Multiplies each task across all configured orchestration modes and rewrites `coordinate_mode` (or configured orchestration key) in temp configs.

## Example workflow: Testing 3 model sizes

Add to manifest:
```yaml
model_sets:
  tiny:
    description: "Tiny models for rapid testing"
    models:
      - openai/qwen2.5:0.5b
      - openai/phi3:3.8b

  small:
    description: "Small production models"
    models:
      - openai/llama2:7b
      - openai/mistral:7b

  medium:
    description: "Medium-sized models"
    models:
      - openai/llama2:13b
      - openai/mistral:8x7b
```

Then run:
```bash
# Run tiny models through all scenarios
python scripts/pipeline/run_ollama_pipeline.py --model-set tiny --dry-run
python scripts/pipeline/run_ollama_pipeline.py --model-set tiny

# Run small models through selected scenarios
python scripts/pipeline/run_ollama_pipeline.py --model-set small --scenarios reasoning,coding

# Run medium models with custom config overrides
python scripts/pipeline/run_ollama_pipeline.py --model-set medium --set max_iterations=5
```

## Notes

- For LiteLLM + Ollama, model names are often used with `openai/` prefix (for example `openai/qwen2.5:0.5b`) when `OPENAI_API_BASE` points to Ollama.
- The default manifest includes a werewolf example that uses raw `model_name` (for example `FunctionGemma`).
- Temporary configs are isolated from source configs, so this pipeline does not modify files in `marble/configs/`.
- Each pipeline run gets a timestamped output folder, so consecutive runs don't overwrite results.
