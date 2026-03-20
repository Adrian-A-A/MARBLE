#!/usr/bin/env python3
"""Run MARBLE scenarios across multiple Ollama models from one pipeline manifest."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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

    # Common schema: metrics.evaluate_llm can be either a string or a dict with a model key.
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
    output_root_base = repo_root / (global_section.get("output_root") or "logs/pipeline")
    temp_root_base = repo_root / (global_section.get("temp_config_root") or "logs/pipeline_tmp")
    output_root = output_root_base / timestamp
    temp_root = temp_root_base / timestamp

    # Resolve models: CLI --models takes precedence, then --model-set, then scenario-specific.
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

        # Use resolved models, scenario-specific models, or fail.
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
                raise ValueError(
                    f"Scenario '{scenario_name}' has invalid orchestration modes."
                )
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

    # Route output files into per-run folders to avoid collisions between model runs.
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
        # Prefer module execution so package imports resolve reliably.
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
        description="Run MARBLE scenarios across multiple Ollama models from one manifest."
    )
    parser.add_argument(
        "--manifest",
        default="scripts/pipeline/ollama_pipeline.yaml",
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
