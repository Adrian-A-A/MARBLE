#!/usr/bin/env python3
"""Export comparison CSV tables from vLLM pipeline evaluation summaries."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def find_run_dirs(logs_root: Path, requested_runs: Optional[List[str]]) -> List[Path]:
    """Resolve run directories that contain evaluation_summary.json."""
    if requested_runs:
        run_dirs: List[Path] = []
        for run in requested_runs:
            candidate = Path(run)
            if candidate.is_absolute():
                run_dir = candidate
            else:
                run_dir = logs_root / run
            if (run_dir / "evaluation_summary.json").exists():
                run_dirs.append(run_dir)
        return sorted(run_dirs)

    if not logs_root.exists():
        return []

    timestamp_pattern = re.compile(r"^\d{8}_\d{6}$")
    run_dirs = [
        p
        for p in logs_root.iterdir()
        if p.is_dir()
        and timestamp_pattern.match(p.name)
        and (p / "evaluation_summary.json").exists()
    ]
    return sorted(run_dirs)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def to_json_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def flatten_run_row(run_dir: Path, summary: Dict[str, Any]) -> Dict[str, str]:
    aggregate_raw = summary.get("aggregate")
    aggregate: Dict[str, Any] = aggregate_raw if isinstance(aggregate_raw, dict) else {}

    existing_raw = summary.get("existing_script_evaluations")
    existing: Dict[str, Any] = existing_raw if isinstance(existing_raw, dict) else {}

    db_eval_raw = existing.get("database_batch_eval")
    db_eval: Dict[str, Any] = db_eval_raw if isinstance(db_eval_raw, dict) else {}
    db_status = db_eval.get("status", "")

    return {
        "run_id": run_dir.name,
        "evaluation_summary": str((run_dir / "evaluation_summary.json").as_posix()),
        "generated_at": str(summary.get("generated_at", "")),
        "total_tasks": str(aggregate.get("total_tasks", "")),
        "passed": str(aggregate.get("passed", "")),
        "failed": str(aggregate.get("failed", "")),
        "skipped": str(aggregate.get("skipped", "")),
        "output_found": str(aggregate.get("output_found", "")),
        "tasks_with_existing_evaluations": str(
            aggregate.get("tasks_with_existing_evaluations", "")
        ),
        "tasks_by_scenario": to_json_cell(aggregate.get("tasks_by_scenario")),
        "db_batch_eval_status": str(db_status or ""),
    }


def flatten_task_rows(run_dir: Path, summary: Dict[str, Any]) -> Iterable[Dict[str, str]]:
    tasks = summary.get("tasks")
    if not isinstance(tasks, list):
        return []

    rows: List[Dict[str, str]] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue

        rows.append(
            {
                "run_id": run_dir.name,
                "scenario": str(task.get("scenario", "")),
                "model": str(task.get("model", "")),
                "orchestration_mode": str(task.get("orchestration_mode", "")),
                "source_config": str(task.get("source_config", "")),
                "status": str(task.get("status", "")),
                "exit_code": to_json_cell(task.get("exit_code")),
                "output_exists": to_json_cell(task.get("output_exists")),
                "output_file": str(task.get("output_file", "")),
                "existing_evaluations": to_json_cell(task.get("existing_evaluations")),
            }
        )

    return rows


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export run-level and task-level comparison CSV tables from vLLM pipeline evaluation summaries."
    )
    parser.add_argument(
        "--runs",
        default="",
        help=(
            "Comma-separated run ids (timestamps) or absolute run paths. "
            "If omitted, all runs with evaluation_summary.json are included."
        ),
    )
    parser.add_argument(
        "--run-csv",
        default="",
        help=(
            "Output path for run-level comparison CSV. "
            "Defaults to logs/pipeline_vllm/comparison/runs_<timestamp>.csv"
        ),
    )
    parser.add_argument(
        "--task-csv",
        default="",
        help=(
            "Optional output path for task-level CSV. "
            "Defaults to logs/pipeline_vllm/comparison/tasks_<timestamp>.csv"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    logs_root = repo_root / "logs" / "pipeline_vllm"

    requested_runs = [x.strip() for x in args.runs.split(",") if x.strip()] or None
    run_dirs = find_run_dirs(logs_root, requested_runs)
    if not run_dirs:
        raise FileNotFoundError(
            "No evaluation summaries found. Run stage 2 first: scripts/pipeline/run_vllm_evaluation_pipeline.py"
        )

    now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_root = logs_root / "comparison"
    run_csv = Path(args.run_csv).resolve() if args.run_csv else comparison_root / f"runs_{now}.csv"
    task_csv = Path(args.task_csv).resolve() if args.task_csv else comparison_root / f"tasks_{now}.csv"

    run_rows: List[Dict[str, str]] = []
    task_rows: List[Dict[str, str]] = []

    for run_dir in run_dirs:
        summary_path = run_dir / "evaluation_summary.json"
        summary = read_json(summary_path)
        run_rows.append(flatten_run_row(run_dir, summary))
        task_rows.extend(flatten_task_rows(run_dir, summary))

    run_fields = [
        "run_id",
        "evaluation_summary",
        "generated_at",
        "total_tasks",
        "passed",
        "failed",
        "skipped",
        "output_found",
        "tasks_with_existing_evaluations",
        "tasks_by_scenario",
        "db_batch_eval_status",
    ]
    task_fields = [
        "run_id",
        "scenario",
        "model",
        "orchestration_mode",
        "source_config",
        "status",
        "exit_code",
        "output_exists",
        "output_file",
        "existing_evaluations",
    ]

    write_csv(run_csv, run_rows, run_fields)
    write_csv(task_csv, task_rows, task_fields)

    print("[COMPARE EXPORT]")
    print(f"  runs analyzed : {len(run_dirs)}")
    print(f"  run csv      : {run_csv}")
    print(f"  task csv     : {task_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
