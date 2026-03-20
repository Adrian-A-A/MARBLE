#!/usr/bin/env python3
"""Evaluate MARBLE pipeline runs from generated task manifests."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def read_jsonl_last(path: Path) -> Optional[Dict[str, Any]]:
    """Read the last valid JSON object from a JSONL file."""
    if not path.exists() or not path.is_file():
        return None

    last_obj: Optional[Dict[str, Any]] = None
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                last_obj = obj

    return last_obj


def extract_existing_evaluations(out_obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract only evaluation fields produced by existing repository evaluators."""
    if not out_obj:
        return {}

    extracted: Dict[str, Any] = {}
    for key in [
        "planning_scores",
        "communication_scores",
        "task_evaluation",
        "code_quality",
        "token_usage",
        "agent_kpis",
        "total_milestones",
    ]:
        if key in out_obj:
            extracted[key] = out_obj[key]
    return extracted


def evaluate_task_record(repo_root: Path, task: Dict[str, Any]) -> Dict[str, Any]:
    """Collect one task record using run metadata and produced JSONL output."""
    output_rel = task.get("expected_output_file")
    run_log_rel = task.get("run_log")

    output_path = repo_root / output_rel if output_rel else None
    run_log_path = repo_root / run_log_rel if run_log_rel else None

    status = task.get("status")
    exit_code = task.get("exit_code")
    scenario = task.get("scenario")

    out_obj = read_jsonl_last(output_path) if output_path else None
    existing_eval = extract_existing_evaluations(out_obj)

    return {
        "scenario": scenario,
        "model": task.get("model"),
        "orchestration_mode": task.get("orchestration_mode"),
        "source_config": task.get("source_config"),
        "status": status,
        "exit_code": exit_code,
        "run_log_exists": bool(run_log_path and run_log_path.exists()),
        "output_exists": bool(output_path and output_path.exists()),
        "output_file": str(output_path.relative_to(repo_root).as_posix())
        if output_path is not None
        else None,
        "existing_evaluations": existing_eval,
    }


def prepare_db_batch_eval_inputs(
    repo_root: Path, run_root: Path, task_results: List[Dict[str, Any]]
) -> Optional[Path]:
    """Create folder structure expected by scripts/database/batch_eval.py from DB outputs."""
    db_tasks = [r for r in task_results if r.get("scenario") == "database"]
    if not db_tasks:
        return None

    input_root = run_root / "existing_eval" / "db_batch_eval_input"
    input_root.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in db_tasks:
        model = str(rec.get("model") or "unknown_model")
        group_name = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_") or "db"
        grouped.setdefault(group_name, []).append(rec)

    for group_name, records in grouped.items():
        group_dir = input_root / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        for idx, rec in enumerate(records, start=1):
            output_file = rec.get("output_file")
            if not output_file:
                continue
            out_obj = read_jsonl_last(repo_root / output_file)
            if not isinstance(out_obj, dict):
                continue
            json_path = group_dir / f"task_{idx}.json"
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(out_obj, f, indent=2, ensure_ascii=False)

    return input_root


def run_existing_db_batch_eval(
    repo_root: Path, run_root: Path, task_results: List[Dict[str, Any]], python_bin: str
) -> Dict[str, Any]:
    """Execute existing scripts/database/batch_eval.py and capture its output."""
    input_root = prepare_db_batch_eval_inputs(repo_root, run_root, task_results)
    if input_root is None:
        return {"status": "not_applicable", "reason": "No database tasks in run."}

    batch_eval_script = repo_root / "scripts" / "database" / "batch_eval.py"
    if not batch_eval_script.exists():
        return {
            "status": "error",
            "reason": f"Missing script: {batch_eval_script}",
        }

    proc = subprocess.run(
        [python_bin, str(batch_eval_script)],
        cwd=str(input_root),
        capture_output=True,
        text=True,
        check=False,
    )

    return {
        "status": "ok" if proc.returncode == 0 else "error",
        "return_code": proc.returncode,
        "cwd": str(input_root),
        "script": str(batch_eval_script),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def choose_run_root(repo_root: Path, requested: str, manifest_name: str) -> Path:
    """Resolve run root from explicit path, run id, or latest run marker."""
    logs_root = repo_root / "logs" / "pipeline"

    if requested:
        requested_path = Path(requested)
        if requested_path.is_absolute():
            candidate = requested_path
        else:
            candidate = logs_root / requested
        return candidate.resolve()

    if not logs_root.exists():
        raise FileNotFoundError("No pipeline runs found under logs/pipeline.")

    timestamp_pattern = r"^\d{8}_\d{6}$"
    candidates = [
        p
        for p in logs_root.iterdir()
        if p.is_dir()
        and (p / manifest_name).exists()
        and re.match(timestamp_pattern, p.name)
    ]

    if not candidates:
        # Backward-compatible fallback: any folder containing the manifest.
        candidates = [p for p in logs_root.iterdir() if p.is_dir() and (p / manifest_name).exists()]

    if not candidates:
        raise FileNotFoundError("No pipeline run directories found under logs/pipeline.")

    return sorted(candidates)[-1]


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate status counts and presence of existing evaluator outputs."""
    total = len(results)
    passed = sum(1 for r in results if r.get("status") == "ok")
    failed = sum(1 for r in results if r.get("status") == "failed")
    skipped = sum(1 for r in results if r.get("status") == "skipped")

    with_eval = sum(1 for r in results if r.get("existing_evaluations"))
    by_scenario: Dict[str, int] = {}
    for r in results:
        scenario = str(r.get("scenario") or "unknown")
        by_scenario[scenario] = by_scenario.get(scenario, 0) + 1

    return {
        "total_tasks": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "output_found": sum(1 for r in results if r.get("output_exists")),
        "tasks_with_existing_evaluations": with_eval,
        "tasks_by_scenario": by_scenario,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate completed MARBLE pipeline runs from task manifests."
    )
    parser.add_argument(
        "--run",
        default="",
        help=(
            "Run timestamp folder under logs/pipeline (e.g., 20260320_120737), "
            "or an absolute run directory path. If omitted, uses latest."
        ),
    )
    parser.add_argument(
        "--manifest",
        default="tasks_manifest.json",
        help="Manifest filename in run folder.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON path for evaluation summary. Defaults to <run>/evaluation_summary.json",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to run existing evaluation scripts.",
    )
    parser.add_argument(
        "--fail-on-missing-output",
        action="store_true",
        help="Exit non-zero if any successful task has no output file.",
    )
    parser.add_argument(
        "--skip-db-batch-eval",
        action="store_true",
        help="Skip invoking scripts/database/batch_eval.py for database tasks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    run_root = choose_run_root(repo_root, args.run, args.manifest)

    manifest_path = run_root / args.manifest
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Manifest has no tasks to evaluate.")

    results = [evaluate_task_record(repo_root, t) for t in tasks if isinstance(t, dict)]
    aggregate = summarize(results)

    db_batch_eval_result = (
        {"status": "skipped", "reason": "--skip-db-batch-eval specified."}
        if args.skip_db_batch_eval
        else run_existing_db_batch_eval(
            repo_root=repo_root,
            run_root=run_root,
            task_results=results,
            python_bin=args.python_bin,
        )
    )

    output_path = Path(args.output).resolve() if args.output else run_root / "evaluation_summary.json"
    payload = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run_root": str(run_root),
        "manifest": str(manifest_path),
        "aggregate": aggregate,
        "existing_script_evaluations": {
            "database_batch_eval": db_batch_eval_result,
        },
        "tasks": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("[EVAL PIPELINE]")
    print(f"  run      : {run_root}")
    print(f"  manifest : {manifest_path}")
    print(f"  output   : {output_path}")
    print("[EVAL SUMMARY]")
    print(f"  total    : {aggregate['total_tasks']}")
    print(f"  passed   : {aggregate['passed']}")
    print(f"  failed   : {aggregate['failed']}")
    print(f"  skipped  : {aggregate['skipped']}")
    print(f"  outputs  : {aggregate['output_found']}")
    print(
        f"  existing : {aggregate['tasks_with_existing_evaluations']} task(s) with existing evaluator payloads"
    )

    db_status = db_batch_eval_result.get("status")
    if db_status == "ok":
        print("  db_eval  : scripts/database/batch_eval.py executed")
    elif db_status == "not_applicable":
        print("  db_eval  : not applicable (no database tasks)")
    elif db_status == "skipped":
        print("  db_eval  : skipped")
    else:
        print("  db_eval  : failed (see evaluation_summary.json for stdout/stderr)")

    if args.fail_on_missing_output:
        missing = [
            r
            for r in results
            if r.get("status") == "ok" and not r.get("output_exists")
        ]
        if missing:
            print(
                f"  error    : {len(missing)} successful task(s) missing output files"
            )
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
