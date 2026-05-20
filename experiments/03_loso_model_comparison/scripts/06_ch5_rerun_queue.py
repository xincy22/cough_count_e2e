"""
Single mainline runner for the Chapter 5 0.7M rerun.

The script keeps the real remote workflow explicit:

1. audit  - print the exact queue and validate config/model ids.
2. run    - run one shard of the queue on one GPU.
3. report - collect completed loso_summary.json files into CSV tables.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


EXP_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PLAN = EXP_DIR / "configs" / "ch5_rerun_jobs_0p7m.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_plan(path: Path) -> dict[str, Any]:
    plan = load_yaml(path)
    if "jobs" not in plan or not isinstance(plan["jobs"], list):
        raise ValueError(f"Invalid plan without jobs list: {path}")
    return plan


def config_model_ids(config_path: Path) -> set[str]:
    cfg = load_yaml(config_path)
    return {str(m["id"]) for m in cfg.get("models", [])}


def _repo_root() -> Path:
    return EXP_DIR.parents[1]


def _find_model_cfg(config_path: Path, model_id: str) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    for model in cfg.get("models", []):
        if str(model.get("id")) == str(model_id):
            return model
    raise KeyError(f"model_id={model_id} not found in {config_path}")


def actual_trainable_params(config_path: Path, model_id: str) -> int:
    repo_root = _repo_root()
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from coughcount.models.builder import build_model

    model_cfg = _find_model_cfg(config_path, model_id)
    arch_type = str(model_cfg["architecture"]["type"])
    cfg = {
        "model": {
            "name": arch_type,
            "presets": {
                arch_type: model_cfg["architecture"],
            },
        }
    }
    model = build_model(cfg, in_channels=513)
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def validate_plan(plan: dict[str, Any]) -> None:
    seen: set[str] = set()
    for job in plan["jobs"]:
        job_id = str(job["job_id"])
        if job_id in seen:
            raise ValueError(f"Duplicate job_id: {job_id}")
        seen.add(job_id)
        config_path = EXP_DIR / str(job["config"])
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config for {job_id}: {config_path}")
        ids = config_model_ids(config_path)
        if str(job["model_id"]) not in ids:
            raise ValueError(
                f"{job_id} model_id={job['model_id']} not found in {config_path.name}"
            )
        actual = actual_trainable_params(config_path, str(job["model_id"]))
        expected = int(job["expected_params"])
        if actual != expected:
            raise ValueError(
                f"{job_id} expected_params={expected} but actual={actual}; "
                f"fix configs/ch5_rerun_jobs_0p7m.yaml or {config_path.name}"
            )


def selected_jobs(
    plan: dict[str, Any],
    *,
    shard_index: int | None,
    num_shards: int | None,
    group: str | None,
    job_ids: set[str] | None,
) -> list[dict[str, Any]]:
    jobs = list(plan["jobs"])
    if group:
        jobs = [j for j in jobs if str(j.get("group")) == group]
    if job_ids:
        jobs = [j for j in jobs if str(j.get("job_id")) in job_ids]
    if shard_index is not None:
        n = int(num_shards or plan.get("default_num_shards", 1))
        jobs = [j for j in jobs if int(j.get("shard", -1)) % n == int(shard_index)]
    return jobs


def print_audit(plan: dict[str, Any], jobs: list[dict[str, Any]]) -> None:
    validate_plan(plan)
    print(f"Plan: {plan.get('name')}")
    print(f"Jobs selected: {len(jobs)} / {len(plan['jobs'])}")
    print()
    print(
        f"{'job':<5} {'shard':<5} {'group':<10} {'model_id':<8} "
        f"{'model_name':<22} {'params':>10} {'actual':>10} config"
    )
    print("-" * 112)
    for job in jobs:
        actual = actual_trainable_params(EXP_DIR / str(job["config"]), str(job["model_id"]))
        print(
            f"{job['job_id']:<5} {job['shard']:<5} {job['group']:<10} "
            f"{job['model_id']:<8} {job['model_name']:<22} "
            f"{int(job['expected_params']):>10} {actual:>10} {job['config']}"
        )


def stream_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("COMMAND:", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8", newline="") as log:
        proc = subprocess.Popen(
            command,
            cwd=EXP_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return int(proc.wait())


def run_jobs(
    plan: dict[str, Any],
    jobs: list[dict[str, Any]],
    *,
    device: str,
    batch_size: int | None,
    num_workers: int | None,
    dry_run: bool,
) -> None:
    validate_plan(plan)
    if not jobs:
        raise SystemExit("No jobs selected.")

    bs = int(batch_size or plan.get("default_batch_size", 24))
    nw = int(num_workers if num_workers is not None else plan.get("default_num_workers", 4))
    log_dir = EXP_DIR / "runs" / "ch5_rerun_0p7m_logs"

    for job in jobs:
        command = [
            sys.executable,
            str(EXP_DIR / "scripts" / "03_loso.py"),
            "--config",
            str(job["config"]),
            "--model-id",
            str(job["model_id"]),
            "--device",
            str(device),
            "--batch-size",
            str(bs),
            "--num-workers",
            str(nw),
        ]
        log_path = log_dir / f"{job['job_id']}_{job['model_name']}.log"
        print(f"\n=== {job['job_id']} {job['model_name']} on {device} ===", flush=True)
        if dry_run:
            print("DRY RUN:", " ".join(command))
            continue
        code = stream_command(command, log_path)
        if code != 0:
            raise SystemExit(f"Job {job['job_id']} failed with exit code {code}")


def latest_summary_for_model(exp_name: str, model_id: str) -> Path | None:
    candidates = sorted(
        EXP_DIR.glob(f"runs/{exp_name}_*/{model_id}/loso_summary.json"),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def report(plan: dict[str, Any]) -> Path:
    validate_plan(plan)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = EXP_DIR / "result" / f"ch5_rerun_0p7m_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for job in plan["jobs"]:
        cfg = load_yaml(EXP_DIR / str(job["config"]))
        exp_name = str(cfg.get("loso", {}).get("exp_name", cfg.get("name")))
        summary_path = latest_summary_for_model(exp_name, str(job["model_id"]))
        if summary_path is None:
            missing.append(job)
            continue
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "job_id": job["job_id"],
                "group": job["group"],
                "model_id": job["model_id"],
                "model_name": job["model_name"],
                "expected_params": int(job["expected_params"]),
                "num_folds": int(data["num_folds"]),
                "mean_count_mae": float(data["mean_count_mae"]),
                "std_count_mae": float(data["std_count_mae"]),
                "mean_count_mae_pos": float(data["mean_count_mae_pos"]),
                "std_count_mae_pos": float(data["std_count_mae_pos"]),
                "mean_count_mae_neg": float(data["mean_count_mae_neg"]),
                "std_count_mae_neg": float(data["std_count_mae_neg"]),
                "summary_json": str(summary_path.relative_to(EXP_DIR)),
            }
        )

    fields = [
        "job_id",
        "group",
        "model_id",
        "model_name",
        "expected_params",
        "num_folds",
        "mean_count_mae",
        "std_count_mae",
        "mean_count_mae_pos",
        "std_count_mae_pos",
        "mean_count_mae_neg",
        "std_count_mae_neg",
        "summary_json",
    ]
    for name, subset in [
        ("all_summary.csv", rows),
        ("structure_summary.csv", [r for r in rows if r["group"] == "structure"]),
        ("ablation_summary.csv", [r for r in rows if r["group"] == "ablation"]),
    ]:
        with (out_dir / name).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(subset)

    with (out_dir / "missing_jobs.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["job_id", "group", "model_id", "model_name", "config"],
        )
        writer.writeheader()
        writer.writerows(missing)

    lines = [
        "# Chapter 5 Rerun Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Completed jobs: {len(rows)} / {len(plan['jobs'])}",
        f"Missing jobs: {len(missing)}",
        "",
    ]
    if rows:
        lines.extend(
            [
                "| Group | Model | Params | Folds | Count MAE | Pos MAE | Neg MAE |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in sorted(rows, key=lambda r: (r["group"], r["job_id"])):
            lines.append(
                f"| {row['group']} | {row['model_name']} | {row['expected_params']} | "
                f"{row['num_folds']} | {row['mean_count_mae']:.4f} ± {row['std_count_mae']:.4f} | "
                f"{row['mean_count_mae_pos']:.4f} ± {row['std_count_mae_pos']:.4f} | "
                f"{row['mean_count_mae_neg']:.4f} ± {row['std_count_mae_neg']:.4f} |"
            )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote report to {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["audit", "run", "report"])
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--group", choices=["structure", "ablation"], default=None)
    parser.add_argument("--job-id", action="append", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = load_plan(args.plan)
    job_ids = set(args.job_id) if args.job_id else None
    jobs = selected_jobs(
        plan,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        group=args.group,
        job_ids=job_ids,
    )

    if args.command == "audit":
        print_audit(plan, jobs)
    elif args.command == "run":
        run_jobs(
            plan,
            jobs,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            dry_run=args.dry_run,
        )
    elif args.command == "report":
        report(plan)


if __name__ == "__main__":
    main()
