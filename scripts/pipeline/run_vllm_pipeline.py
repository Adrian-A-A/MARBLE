#!/usr/bin/env python3
"""Run MARBLE scenarios across multiple vLLM-backed models from one pipeline manifest."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import gc
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib import request as url_request

import yaml


DEFAULT_MODEL_KEYS = ["llm", "model", "model_name", "evaluate_llm"]
DEFAULT_ORCHESTRATION_KEY = "coordinate_mode"
DEFAULT_ORCHESTRATION_MODES = ["star", "graph", "chain", "tree"]


def safe_slug(value: str) -> str:
    """Create a filesystem-safe slug from an arbitrary string."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "default"


def parse_key_value_args(items: Sequence[str]) -> Dict[str, str]:
    """Parse KEY=VALUE CLI arguments into a dictionary."""
    parsed: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --set value '{item}'. Expected KEY=VALUE format.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --set value '{item}'. Key cannot be empty.")
        parsed[key] = value
    return parsed


def deep_set_key(obj: Any, target_key: str, new_value: Any) -> None:
    """Recursively replace all values under a given key in nested dict/list structures."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == target_key:
                obj[key] = new_value
            else:
                deep_set_key(value, target_key, new_value)
    elif isinstance(obj, list):
        for item in obj:
            deep_set_key(item, target_key, new_value)


def apply_model_to_config(config: Dict[str, Any], model: str, model_keys: Sequence[str]) -> Dict[str, Any]:
    """Return a copy of config with all target model keys updated to a model string."""
    updated = copy.deepcopy(config)

    for key in model_keys:
        deep_set_key(updated, key, model)

    metrics = updated.get("metrics")
    if isinstance(metrics, dict) and "evaluate_llm" in metrics:
        eval_llm = metrics["evaluate_llm"]
        if isinstance(eval_llm, str):
            metrics["evaluate_llm"] = model
        elif isinstance(eval_llm, dict) and "model" in eval_llm:
            eval_llm["model"] = model

    return updated


def discover_configs(repo_root: Path, patterns: Sequence[str]) -> List[Path]:
    """Expand glob patterns into a sorted list of unique config paths."""
    discovered: List[Path] = []
    seen: set[str] = set()

    for pattern in patterns:
        for path in repo_root.glob(pattern):
            if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}:
                resolved = str(path.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    discovered.append(path)

    return sorted(discovered)


@dataclass
class RunTask:
    scenario: str
    model: str
    orchestration_mode: Optional[str]
    orchestration_key: str
    source_config: Path
    temp_config: Path
    output_dir: Path


@dataclass
class VLLMServerOptions:
    enabled: bool
    binary: str
    host: str
    port: int
    startup_timeout: int
    device: str
    trust_remote_code: bool
    extra_args: str
    hf_token: str
    hf_cache_dir: str
    api_key: str
    clear_vram_after_model: bool
    clear_hf_cache_after_model: bool


def build_tasks(
    repo_root: Path,
    manifest: Dict[str, Any],
    selected_scenarios: Optional[set[str]],
    selected_models: Optional[List[str]],
    model_set_name: Optional[str],
    orchestration_scope: str,
    timestamp: str,
) -> Tuple[List[RunTask], Dict[str, Any]]:
    """Create concrete run tasks from manifest entries and optional CLI filters."""
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError("Manifest must define a non-empty 'scenarios' mapping.")

    global_section = manifest.get("global") or {}
    global_model_keys = global_section.get("model_keys") or DEFAULT_MODEL_KEYS
    output_root_base = repo_root / (global_section.get("output_root") or "logs/pipeline_vllm")
    temp_root_base = repo_root / (global_section.get("temp_config_root") or "logs/pipeline_tmp_vllm")
    output_root = output_root_base / timestamp
    temp_root = temp_root_base / timestamp

    resolved_models: Optional[List[str]] = None
    if selected_models is not None:
        resolved_models = selected_models
    elif model_set_name:
        model_sets = manifest.get("model_sets") or {}
        if model_set_name not in model_sets:
            raise ValueError(
                f"Model set '{model_set_name}' not found. Available: {', '.join(model_sets.keys())}"
            )
        model_set_cfg = model_sets[model_set_name]
        if not isinstance(model_set_cfg, dict):
            raise ValueError(f"Model set '{model_set_name}' must be a mapping.")
        resolved_models = model_set_cfg.get("models") or []
        if not resolved_models:
            raise ValueError(f"Model set '{model_set_name}' has no models defined.")

    tasks: List[RunTask] = []

    for scenario_name, scenario_cfg in scenarios.items():
        if selected_scenarios and scenario_name not in selected_scenarios:
            continue

        if not isinstance(scenario_cfg, dict):
            raise ValueError(f"Scenario '{scenario_name}' must be a mapping.")

        patterns = scenario_cfg.get("configs") or []
        if not patterns:
            raise ValueError(f"Scenario '{scenario_name}' has no config patterns in 'configs'.")

        configs = discover_configs(repo_root, patterns)
        if not configs:
            print(f"[WARN] Scenario '{scenario_name}' matched no configs for patterns: {patterns}")
            continue

        max_configs = scenario_cfg.get("max_configs")
        if isinstance(max_configs, int) and max_configs > 0:
            configs = configs[:max_configs]

        models = resolved_models or scenario_cfg.get("models")
        if not models:
            raise ValueError(
                f"Scenario '{scenario_name}' has no models. Provide --models, --model-set, or define scenario models."
            )

        model_keys = scenario_cfg.get("model_keys") or global_model_keys
        orchestration_key = (
            scenario_cfg.get("orchestration_key")
            or global_section.get("orchestration_key")
            or DEFAULT_ORCHESTRATION_KEY
        )

        if orchestration_scope == "all":
            orchestration_modes = (
                scenario_cfg.get("orchestration_modes")
                or global_section.get("orchestration_modes")
                or DEFAULT_ORCHESTRATION_MODES
            )
            if not isinstance(orchestration_modes, list) or not orchestration_modes:
                raise ValueError(f"Scenario '{scenario_name}' has invalid orchestration modes.")
        else:
            orchestration_modes = [None]

        for model in models:
            model_slug = safe_slug(model)
            for orchestration_mode in orchestration_modes:
                mode_slug = safe_slug(orchestration_mode) if orchestration_mode else None
                for cfg_path in configs:
                    rel_cfg = cfg_path.relative_to(repo_root)
                    cfg_slug = safe_slug(str(rel_cfg.with_suffix("")))

                    if mode_slug:
                        temp_config = temp_root / scenario_name / model_slug / mode_slug / f"{cfg_slug}.yaml"
                        output_dir = output_root / scenario_name / model_slug / mode_slug / cfg_slug
                    else:
                        temp_config = temp_root / scenario_name / model_slug / f"{cfg_slug}.yaml"
                        output_dir = output_root / scenario_name / model_slug / cfg_slug

                    tasks.append(
                        RunTask(
                            scenario=scenario_name,
                            model=model,
                            orchestration_mode=orchestration_mode,
                            orchestration_key=orchestration_key,
                            source_config=cfg_path,
                            temp_config=temp_config,
                            output_dir=output_dir,
                        )
                    )

    runtime = {
        "global": global_section,
        "output_root": output_root,
        "temp_root": temp_root,
        "timestamp": timestamp,
    }
    return tasks, runtime


def write_temp_config(
    task: RunTask,
    model_keys: Sequence[str],
    extra_set_values: Dict[str, str],
) -> Optional[Path]:
    """Create a per-run config file with model fields (and optional keys) overridden."""
    task.temp_config.parent.mkdir(parents=True, exist_ok=True)
    task.output_dir.mkdir(parents=True, exist_ok=True)

    with task.source_config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config {task.source_config} does not contain a YAML mapping.")

    updated = apply_model_to_config(config, task.model, model_keys)

    if task.orchestration_mode:
        deep_set_key(updated, task.orchestration_key, task.orchestration_mode)

    expected_output_file: Optional[Path] = None
    output = updated.get("output")
    if isinstance(output, dict) and output.get("file_path"):
        output_name = Path(str(output["file_path"])).name
        expected_output_file = task.output_dir / output_name
        output["file_path"] = str(expected_output_file.as_posix())

    for key, value in extra_set_values.items():
        deep_set_key(updated, key, value)

    with task.temp_config.open("w", encoding="utf-8") as f:
        yaml.safe_dump(updated, f, sort_keys=False, allow_unicode=False)

    return expected_output_file


def serialize_task_record(
    repo_root: Path,
    task: RunTask,
    expected_output_file: Optional[Path],
) -> Dict[str, Any]:
    """Convert a task and resolved paths into a JSON-safe manifest record."""
    run_log = task.output_dir / "run.log"
    record: Dict[str, Any] = {
        "scenario": task.scenario,
        "model": task.model,
        "orchestration_mode": task.orchestration_mode or "default",
        "source_config": str(task.source_config.relative_to(repo_root).as_posix()),
        "temp_config": str(task.temp_config.relative_to(repo_root).as_posix()),
        "output_dir": str(task.output_dir.relative_to(repo_root).as_posix()),
        "run_log": str(run_log.relative_to(repo_root).as_posix()),
        "status": "pending",
        "exit_code": None,
    }
    if expected_output_file is not None:
        record["expected_output_file"] = str(
            expected_output_file.relative_to(repo_root).as_posix()
        )
    else:
        record["expected_output_file"] = None
    return record


def write_run_manifest(manifest_path: Path, payload: Dict[str, Any]) -> None:
    """Persist pipeline run metadata for downstream evaluation."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def strip_openai_prefix(model: str) -> str:
    """Return an HF-style model id by removing the optional openai/ prefix."""
    if model.startswith("openai/"):
        return model[len("openai/") :]
    return model


def wait_for_vllm_ready(api_base: str, api_key: str, timeout_sec: int) -> None:
    """Poll vLLM OpenAI-compatible endpoint until model list is reachable."""
    deadline = time.time() + timeout_sec
    url = api_base.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    last_error: Optional[str] = None
    while time.time() < deadline:
        req = url_request.Request(url, headers=headers)
        try:
            with url_request.urlopen(req, timeout=5) as resp:
                if 200 <= resp.status < 300:
                    return
                last_error = f"HTTP {resp.status}"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(2)

    raise TimeoutError(
        f"Timed out waiting for vLLM server at {url}. Last error: {last_error or 'unknown'}"
    )


def wait_for_vllm_chat_ready(
    api_base: str,
    api_key: str,
    model: str,
    timeout_sec: int,
) -> None:
    """Poll chat completions endpoint until the served model can answer a tiny request."""
    deadline = time.time() + timeout_sec
    url = api_base.rstrip("/") + "/chat/completions"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0,
    }

    last_error: Optional[str] = None
    while time.time() < deadline:
        req = url_request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with url_request.urlopen(req, timeout=20) as resp:
                if 200 <= resp.status < 300:
                    return
                last_error = f"HTTP {resp.status}"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(2)

    raise TimeoutError(
        f"Timed out waiting for vLLM chat readiness at {url}. "
        f"Last error: {last_error or 'unknown'}"
    )


def build_vllm_serve_command(model: str, opts: VLLMServerOptions) -> List[str]:
    """Build `vllm serve` command for one model."""
    command = [
        opts.binary,
        "serve",
        model,
        "--host",
        opts.host,
        "--port",
        str(opts.port),
    ]
    if opts.device and opts.device != "auto":
        command.extend(["--device", opts.device])
    if opts.trust_remote_code:
        command.append("--trust-remote-code")
    if opts.api_key:
        command.extend(["--api-key", opts.api_key])
    if opts.extra_args:
        command.extend(shlex.split(opts.extra_args))
    return command


def stop_vllm_server(process: subprocess.Popen[Any], timeout: int = 30) -> None:
    """Terminate vLLM server process gracefully, then force-kill if needed."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def clear_vram_best_effort() -> None:
    """Best-effort VRAM cleanup after terminating vLLM process."""
    gc.collect()
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        # Optional cleanup only; process teardown already releases model VRAM.
        pass


def resolve_hf_hub_cache_dir(explicit_cache_dir: str, env: Dict[str, str]) -> Path:
    """Resolve Hugging Face hub cache root path from CLI/env/defaults."""
    if explicit_cache_dir:
        return Path(explicit_cache_dir).expanduser()
    if env.get("HUGGINGFACE_HUB_CACHE"):
        return Path(env["HUGGINGFACE_HUB_CACHE"]).expanduser()
    if env.get("HF_HUB_CACHE"):
        return Path(env["HF_HUB_CACHE"]).expanduser()
    if env.get("HF_HOME"):
        return Path(env["HF_HOME"]).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def cleanup_model_hf_cache(model_id: str, opts: VLLMServerOptions, env: Dict[str, str]) -> None:
    """Delete one model repo from HF hub cache to bound disk usage."""
    cache_root = resolve_hf_hub_cache_dir(opts.hf_cache_dir, env)
    repo_dir = cache_root / f"models--{model_id.replace('/', '--')}"

    if not repo_dir.exists():
        print(f"[HF CACHE] no cache dir found for {model_id} under {cache_root}")
        return

    try:
        shutil.rmtree(repo_dir)
        print(f"[HF CACHE] removed {repo_dir}")
    except Exception as exc:  # noqa: BLE001
        print(f"[HF CACHE] failed to remove {repo_dir}: {exc}")


def run_task(
    repo_root: Path,
    python_bin: str,
    main_script: str,
    task: RunTask,
    env_overrides: Dict[str, str],
    dry_run: bool,
) -> int:
    """Execute one run task and return the process exit code."""
    main_path = Path(main_script)
    default_main = (repo_root / "marble" / "main.py").resolve()
    run_cwd = repo_root
    if main_path.resolve() == default_main:
        command = [python_bin, "-m", "marble.main", "--config_path", str(task.temp_config)]
    else:
        command = [python_bin, main_script, "--config_path", str(task.temp_config)]

    print("\n[RUN]")
    print(f"  scenario : {task.scenario}")
    print(f"  model    : {task.model}")
    print(f"  orch     : {task.orchestration_mode or 'default(config)'}")
    print(f"  config   : {task.source_config.relative_to(repo_root)}")
    print(f"  temp_cfg : {task.temp_config.relative_to(repo_root)}")
    print(f"  out_dir  : {task.output_dir.relative_to(repo_root)}")
    print(f"  command  : {' '.join(command)}")

    if dry_run:
        return 0

    run_env = os.environ.copy()
    run_env.update(env_overrides)
    existing_pythonpath = run_env.get("PYTHONPATH", "")
    run_env["PYTHONPATH"] = (
        str(repo_root)
        if not existing_pythonpath
        else str(repo_root) + os.pathsep + existing_pythonpath
    )
    run_env["MARBLE_PIPELINE_SCENARIO"] = task.scenario
    run_env["MARBLE_PIPELINE_MODEL"] = task.model
    run_env["MARBLE_PIPELINE_ORCHESTRATION"] = task.orchestration_mode or "default"
    run_env["MARBLE_PIPELINE_OUTPUT_DIR"] = str(task.output_dir)

    log_file = task.output_dir / "run.log"
    with log_file.open("w", encoding="utf-8") as log_f:
        process = subprocess.run(
            command,
            cwd=run_cwd,
            env=run_env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )

    print(f"  status   : {'ok' if process.returncode == 0 else 'failed'}")
    print(f"  run.log  : {log_file.relative_to(repo_root)}")
    return process.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MARBLE scenarios across multiple vLLM models from one manifest."
    )
    parser.add_argument(
        "--manifest",
        default="scripts/pipeline/vllm_pipeline.yaml",
        help="Path to pipeline manifest YAML.",
    )
    parser.add_argument(
        "--model-set",
        default="",
        help="Run all scenarios with all models from a named model set (e.g., tiny, small, medium).",
    )
    parser.add_argument(
        "--orchestration",
        choices=["default", "all"],
        default="default",
        help="Run only the config's default orchestration mode, or sweep all orchestration modes.",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model list to override manifest models (takes precedence over --model-set).",
    )
    parser.add_argument(
        "--scenarios",
        default="",
        help="Comma-separated scenario names to run.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable to run MARBLE.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional recursive key overrides applied to generated configs.",
    )
    parser.add_argument(
        "--api-base",
        default="",
        help="Override OPENAI_API_BASE for this run (for example http://127.0.0.1:8000/v1).",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Override OPENAI_API_KEY for this run.",
    )
    parser.add_argument(
        "--manage-vllm-server",
        action="store_true",
        help=(
            "Start/stop a local vLLM server per model automatically. "
            "Useful for large Hugging Face models that must be loaded one-by-one."
        ),
    )
    parser.add_argument(
        "--vllm-binary",
        default="vllm",
        help="vLLM CLI binary used when --manage-vllm-server is enabled.",
    )
    parser.add_argument(
        "--vllm-host",
        default="127.0.0.1",
        help="Host for managed vLLM server.",
    )
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=8000,
        help="Port for managed vLLM server.",
    )
    parser.add_argument(
        "--vllm-startup-timeout",
        type=int,
        default=240,
        help="Seconds to wait for managed vLLM server readiness.",
    )
    parser.add_argument(
        "--vllm-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device passed to managed vLLM server.",
    )
    parser.add_argument(
        "--vllm-trust-remote-code",
        action="store_true",
        help="Pass --trust-remote-code to managed vLLM server.",
    )
    parser.add_argument(
        "--vllm-extra-args",
        default="",
        help="Extra raw arguments appended to `vllm serve` command.",
    )
    parser.add_argument(
        "--hf-token",
        default="",
        help="Optional Hugging Face token used by managed vLLM server to download gated models.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default="",
        help=(
            "Optional Hugging Face hub cache directory. "
            "Defaults to HUGGINGFACE_HUB_CACHE/HF_HOME or ~/.cache/huggingface/hub."
        ),
    )
    parser.add_argument(
        "--clear-vram-after-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When managing server, clear VRAM caches after each model unload (default: true).",
    )
    parser.add_argument(
        "--clear-hf-cache-after-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When managing server, delete the current model's HF cache directory after it finishes. "
            "Use this when disk space is limited (default: false)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare tasks and print commands without running MARBLE.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately after the first failed run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = (repo_root / args.manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    if not isinstance(manifest, dict):
        raise ValueError("Manifest YAML must contain a top-level mapping.")

    selected_models = [m.strip() for m in args.models.split(",") if m.strip()] or None
    selected_scenarios = (
        {s.strip() for s in args.scenarios.split(",") if s.strip()} or None
    )
    extra_set_values = parse_key_value_args(args.set)
    model_set_name = args.model_set.strip() if args.model_set else None
    orchestration_scope = args.orchestration

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tasks, runtime = build_tasks(
        repo_root=repo_root,
        manifest=manifest,
        selected_scenarios=selected_scenarios,
        selected_models=selected_models,
        model_set_name=model_set_name,
        orchestration_scope=orchestration_scope,
        timestamp=timestamp,
    )

    if not tasks:
        print("No tasks selected. Check --scenarios, --models, --orchestration, and manifest patterns.")
        return 1

    global_cfg = runtime["global"]
    global_model_keys = global_cfg.get("model_keys") or DEFAULT_MODEL_KEYS
    env_overrides = dict(global_cfg.get("env") or {})
    if args.api_base:
        env_overrides["OPENAI_API_BASE"] = args.api_base
    if args.api_key:
        env_overrides["OPENAI_API_KEY"] = args.api_key

    server_api_key = args.api_key or str(env_overrides.get("OPENAI_API_KEY") or "EMPTY")
    managed_api_base = f"http://{args.vllm_host}:{args.vllm_port}/v1"
    vllm_opts = VLLMServerOptions(
        enabled=args.manage_vllm_server,
        binary=args.vllm_binary,
        host=args.vllm_host,
        port=args.vllm_port,
        startup_timeout=args.vllm_startup_timeout,
        device=args.vllm_device,
        trust_remote_code=args.vllm_trust_remote_code,
        extra_args=args.vllm_extra_args,
        hf_token=args.hf_token,
        hf_cache_dir=args.hf_cache_dir,
        api_key=server_api_key,
        clear_vram_after_model=args.clear_vram_after_model,
        clear_hf_cache_after_model=args.clear_hf_cache_after_model,
    )

    main_script = str((repo_root / (global_cfg.get("main_script") or "marble/main.py")).resolve())

    print("[PIPELINE]")
    print(f"  manifest  : {manifest_path.relative_to(repo_root)}")
    print(f"  tasks     : {len(tasks)}")
    print(f"  output    : {runtime['output_root'].relative_to(repo_root)}")
    print(f"  temp_cfg  : {runtime['temp_root'].relative_to(repo_root)}")
    if selected_scenarios:
        print(f"  scenarios : {', '.join(sorted(selected_scenarios))}")
    if model_set_name:
        print(f"  model_set : {model_set_name}")
    if selected_models:
        print(f"  models    : {', '.join(selected_models)}")
    print(f"  orch      : {orchestration_scope}")
    if env_overrides.get("OPENAI_API_BASE"):
        print(f"  api_base  : {env_overrides['OPENAI_API_BASE']}")
    if vllm_opts.enabled:
        print("  vllm_mode : managed-per-model")
        print(f"  vllm_base : {managed_api_base}")
    if args.dry_run:
        print("  mode      : dry-run")

    run_manifest_path = runtime["output_root"] / "tasks_manifest.json"

    failed: List[RunTask] = []
    executed = 0
    task_records: List[Dict[str, Any]] = []

    for task in tasks:
        scenario_cfg = manifest["scenarios"][task.scenario]
        model_keys = scenario_cfg.get("model_keys") or global_model_keys
        expected_output_file = write_temp_config(
            task,
            model_keys=model_keys,
            extra_set_values=extra_set_values,
        )
        task_records.append(
            serialize_task_record(
                repo_root=repo_root,
                task=task,
                expected_output_file=expected_output_file,
            )
        )

    run_payload: Dict[str, Any] = {
        "timestamp": runtime["timestamp"],
        "manifest": str(manifest_path.relative_to(repo_root).as_posix()),
        "python_bin": args.python_bin,
        "orchestration_scope": orchestration_scope,
        "dry_run": args.dry_run,
        "output_root": str(runtime["output_root"].relative_to(repo_root).as_posix()),
        "temp_root": str(runtime["temp_root"].relative_to(repo_root).as_posix()),
        "tasks": task_records,
    }
    write_run_manifest(run_manifest_path, run_payload)

    if vllm_opts.enabled and not args.dry_run:

        model_to_task_indices: Dict[str, List[int]] = {}
        for index, task in enumerate(tasks):
            model_to_task_indices.setdefault(task.model, []).append(index)

        for model_name, indices in model_to_task_indices.items():
            hf_model_id = strip_openai_prefix(model_name)
            model_slug = safe_slug(model_name)
            server_log = runtime["output_root"] / "vllm_server_logs" / f"{model_slug}.log"
            server_log.parent.mkdir(parents=True, exist_ok=True)

            server_env = os.environ.copy()
            if vllm_opts.hf_token:
                server_env["HF_TOKEN"] = vllm_opts.hf_token
                server_env["HUGGING_FACE_HUB_TOKEN"] = vllm_opts.hf_token
            if vllm_opts.hf_cache_dir:
                cache_dir = str(Path(vllm_opts.hf_cache_dir).expanduser())
                server_env["HUGGINGFACE_HUB_CACHE"] = cache_dir
                server_env["HF_HUB_CACHE"] = cache_dir

            serve_cmd = build_vllm_serve_command(hf_model_id, vllm_opts)
            print("\n[VLLM LOAD]")
            print(f"  model    : {model_name}")
            print(f"  hf_model : {hf_model_id}")
            print(f"  command  : {' '.join(serve_cmd)}")
            print(f"  log      : {server_log.relative_to(repo_root)}")

            with server_log.open("w", encoding="utf-8") as log_f:
                server_proc = subprocess.Popen(
                    serve_cmd,
                    cwd=repo_root,
                    env=server_env,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

                server_ready = False
                try:
                    wait_for_vllm_ready(
                        api_base=managed_api_base,
                        api_key=vllm_opts.api_key,
                        timeout_sec=vllm_opts.startup_timeout,
                    )
                    wait_for_vllm_chat_ready(
                        api_base=managed_api_base,
                        api_key=vllm_opts.api_key,
                        model=hf_model_id,
                        timeout_sec=vllm_opts.startup_timeout,
                    )
                    server_ready = True
                except Exception as exc:  # noqa: BLE001
                    print(f"  status   : failed to start ({exc})")

                if not server_ready:
                    stop_vllm_server(server_proc)
                    for index in indices:
                        task = tasks[index]
                        task_records[index]["status"] = "failed"
                        task_records[index]["exit_code"] = 98
                        failed.append(task)
                        executed += 1
                    write_run_manifest(run_manifest_path, run_payload)
                    if args.fail_fast:
                        break
                    continue

                model_env = dict(env_overrides)
                model_env["OPENAI_API_BASE"] = managed_api_base
                model_env["OPENAI_API_KEY"] = vllm_opts.api_key

                for index in indices:
                    task = tasks[index]
                    exit_code = run_task(
                        repo_root=repo_root,
                        python_bin=args.python_bin,
                        main_script=main_script,
                        task=task,
                        env_overrides=model_env,
                        dry_run=False,
                    )
                    executed += 1
                    task_records[index]["status"] = "ok" if exit_code == 0 else "failed"
                    task_records[index]["exit_code"] = exit_code
                    write_run_manifest(run_manifest_path, run_payload)
                    if exit_code != 0:
                        failed.append(task)
                        if args.fail_fast:
                            break

                stop_vllm_server(server_proc)
                if vllm_opts.clear_vram_after_model:
                    clear_vram_best_effort()
                    print("[VLLM UNLOAD] cleared model and ran best-effort VRAM cleanup")
                if vllm_opts.clear_hf_cache_after_model:
                    cleanup_model_hf_cache(hf_model_id, vllm_opts, server_env)

                if args.fail_fast and failed:
                    break
    else:
        for index, task in enumerate(tasks):
            exit_code = run_task(
                repo_root=repo_root,
                python_bin=args.python_bin,
                main_script=main_script,
                task=task,
                env_overrides=env_overrides,
                dry_run=args.dry_run,
            )
            executed += 1
            task_records[index]["status"] = "ok" if exit_code == 0 else "failed"
            task_records[index]["exit_code"] = exit_code
            write_run_manifest(run_manifest_path, run_payload)
            if exit_code != 0:
                failed.append(task)
                if args.fail_fast:
                    break

    if args.fail_fast and executed < len(tasks):
        for index in range(executed, len(tasks)):
            task_records[index]["status"] = "skipped"
            task_records[index]["exit_code"] = None
        write_run_manifest(run_manifest_path, run_payload)

    if args.dry_run:
        executed = len(tasks)

    skipped = len(tasks) - executed

    print("\n[SUMMARY]")
    print(f"  total   : {len(tasks)}")
    print(f"  ran     : {executed}")
    if skipped:
        print(f"  skipped : {skipped}")
    print(f"  failed  : {len(failed)}")
    print(f"  passed  : {executed - len(failed)}")
    print(f"  manifest: {run_manifest_path.relative_to(repo_root)}")

    if failed:
        print("  failed tasks:")
        for task in failed:
            rel_path = task.source_config.relative_to(repo_root)
            print(f"    - {task.scenario} | {task.model} | {rel_path}")
        return 1

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
