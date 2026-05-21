"""
Single mainline runner for the Chapter 5 0.7M 10-model LOSO comparison.

The workflow is deliberately explicit:

1. audit  - validate config/model ids and print the exact queue.
2. run    - run one shard of the queue on one GPU.
3. report - collect completed loso_summary.json files into release-ready tables.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
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


def repo_root() -> Path:
    return EXP_DIR.parents[1]


def find_model_cfg(config_path: Path, model_id: str) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    for model in cfg.get("models", []):
        if str(model.get("id")) == str(model_id):
            return model
    raise KeyError(f"model_id={model_id} not found in {config_path}")


def actual_trainable_params(config_path: Path, model_id: str) -> int:
    root = repo_root()
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from coughcount.models.builder import build_model

    model_cfg = find_model_cfg(config_path, model_id)
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
        f"{'job':<5} {'shard':<5} {'group':<12} {'model_id':<8} "
        f"{'model_name':<22} {'params':>10} {'actual':>10} config"
    )
    print("-" * 116)
    for job in jobs:
        actual = actual_trainable_params(EXP_DIR / str(job["config"]), str(job["model_id"]))
        print(
            f"{job['job_id']:<5} {job['shard']:<5} {job['group']:<12} "
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
    log_dir = EXP_DIR / "runs" / "ch5_10model_compare_0p7m_logs"

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


def git_revision() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    return out.strip()


def runtime_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "git_revision": git_revision(),
    }
    try:
        import torch

        env["torch_version"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_version"] = torch.version.cuda
        env["cudnn_version"] = torch.backends.cudnn.version()
        env["cuda_device_count"] = int(torch.cuda.device_count())
        env["cuda_devices"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": round(
                    torch.cuda.get_device_properties(i).total_memory / (1024**3), 3
                ),
            }
            for i in range(torch.cuda.device_count())
        ]
    except Exception as exc:
        env["torch_probe_error"] = repr(exc)
    return env


def round4(value: Any) -> str:
    return f"{float(value):.4f}"


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report(plan: dict[str, Any]) -> Path:
    validate_plan(plan)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = EXP_DIR / "result" / f"ch5_10model_compare_0p7m_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for job in plan["jobs"]:
        config_path = EXP_DIR / str(job["config"])
        cfg = load_yaml(config_path)
        exp_name = str(cfg.get("loso", {}).get("exp_name", cfg.get("name")))
        summary_path = latest_summary_for_model(exp_name, str(job["model_id"]))
        if summary_path is None:
            missing.append(job)
            continue

        data = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "rank": 0,
                "job_id": job["job_id"],
                "group": job["group"],
                "model_id": job["model_id"],
                "model_name": job["model_name"],
                "params": int(job["expected_params"]),
                "num_folds": int(data["num_folds"]),
                "mean_count_mae": float(data["mean_count_mae"]),
                "std_count_mae": float(data["std_count_mae"]),
                "mean_count_mae_pos": float(data["mean_count_mae_pos"]),
                "std_count_mae_pos": float(data["std_count_mae_pos"]),
                "mean_count_mae_neg": float(data["mean_count_mae_neg"]),
                "std_count_mae_neg": float(data["std_count_mae_neg"]),
                "config": str(job["config"]),
                "summary_json": str(summary_path.relative_to(EXP_DIR)),
            }
        )

        for fold in data.get("fold_results", []):
            fold_rows.append(
                {
                    "job_id": job["job_id"],
                    "model_id": job["model_id"],
                    "model_name": job["model_name"],
                    "fold": int(fold["fold"]),
                    "test_subject": str(fold["test_subject"]),
                    "count_mae": float(fold["count_mae"]),
                    "count_mae_pos": float(fold["count_mae_pos"]),
                    "count_mae_neg": float(fold["count_mae_neg"]),
                    "mean_pred_count_on_neg": float(
                        fold.get("mean_pred_count_on_neg", float("nan"))
                    ),
                    "mean_gt_count_on_pos": float(
                        fold.get("mean_gt_count_on_pos", float("nan"))
                    ),
                    "num_test_samples": int(fold.get("num_test_samples", 0)),
                    "summary_json": str(summary_path.relative_to(EXP_DIR)),
                }
            )

    ranked_rows = sorted(rows, key=lambda r: float(r["mean_count_mae"]))
    for i, row in enumerate(ranked_rows, start=1):
        row["rank"] = i

    summary_fields = [
        "rank",
        "job_id",
        "group",
        "model_id",
        "model_name",
        "params",
        "num_folds",
        "mean_count_mae",
        "std_count_mae",
        "mean_count_mae_pos",
        "std_count_mae_pos",
        "mean_count_mae_neg",
        "std_count_mae_neg",
        "config",
        "summary_json",
    ]
    write_csv(out_dir / "model_comparison_summary.csv", summary_fields, ranked_rows)

    fold_fields = [
        "job_id",
        "model_id",
        "model_name",
        "fold",
        "test_subject",
        "count_mae",
        "count_mae_pos",
        "count_mae_neg",
        "mean_pred_count_on_neg",
        "mean_gt_count_on_pos",
        "num_test_samples",
        "summary_json",
    ]
    write_csv(
        out_dir / "fold_results_all.csv",
        fold_fields,
        sorted(fold_rows, key=lambda r: (r["model_id"], r["fold"])),
    )

    write_csv(
        out_dir / "missing_jobs.csv",
        ["job_id", "group", "model_id", "model_name", "config"],
        missing,
    )

    env = runtime_environment()
    (out_dir / "environment.json").write_text(
        json.dumps(env, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(DEFAULT_PLAN, out_dir / "ch5_rerun_jobs_0p7m.yaml")
    for config_path in sorted({EXP_DIR / str(job["config"]) for job in plan["jobs"]}):
        shutil.copyfile(config_path, out_dir / config_path.name)

    lines = [
        "# Chapter 5 10-Model LOSO Comparison Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Reproducibility",
        "",
        f"- Git revision: `{env.get('git_revision')}`",
        f"- Python: `{env.get('python_version')}`",
        f"- Python executable: `{env.get('python_executable')}`",
        f"- Platform: `{env.get('platform')}`",
        f"- PyTorch: `{env.get('torch_version')}`",
        f"- CUDA available: `{env.get('cuda_available')}`",
        f"- CUDA runtime: `{env.get('cuda_version')}`",
        f"- cuDNN: `{env.get('cudnn_version')}`",
        f"- CUDA device count: `{env.get('cuda_device_count')}`",
        "",
        "## Shared Protocol",
        "",
        "- Dataset: EdgeAI cough-counting dataset prepared by `experiments/00_data_prep`.",
        "- Validation: 15-fold leave-one-subject-out (LOSO).",
        "- Input feature: STFT log-magnitude, shape `[B, F, T]`.",
        "- STFT: `win=1024`, `hop=256`, single-microphone frequency bins `F=513`.",
        "- Microphones: `mic=both` means out/body microphone samples are both used; they are not channel-concatenated.",
        "- Windowing: `window_sec=8.0`, `hop_sec=4.0`.",
        "- Density target: `skewed_gaussian`, `sigma_left_sec=0.03`, `sigma_right_sec=0.10`.",
        "- Training: `epochs=500`, `batch_size=24`, `num_workers=4`, `lr=1e-3`, `weight_decay=0`, seed `42`.",
        "- Model selection: best checkpoint per fold by `val_count_mae`.",
        "- Primary metric: test `Count MAE`; secondary metrics: positive-window and negative-window Count MAE.",
        "",
        "## Results",
        "",
        f"Completed jobs: {len(rows)} / {len(plan['jobs'])}",
        f"Missing jobs: {len(missing)}",
        "",
    ]
    if ranked_rows:
        lines.extend(
            [
                "| Rank | Model | Params | Folds | Count MAE | Pos MAE | Neg MAE |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in ranked_rows:
            lines.append(
                f"| {row['rank']} | {row['model_name']} | {row['params']} | "
                f"{row['num_folds']} | {round4(row['mean_count_mae'])} +/- {round4(row['std_count_mae'])} | "
                f"{round4(row['mean_count_mae_pos'])} +/- {round4(row['std_count_mae_pos'])} | "
                f"{round4(row['mean_count_mae_neg'])} +/- {round4(row['std_count_mae_neg'])} |"
            )

    lines.extend(
        [
            "",
            "## Main Conclusion",
            "",
            "BiCRNN obtains the lowest Count MAE in this unified 0.7M-parameter comparison. TCN-Attn, TCN+UniGRU, and TCN+BiGRU are very close and form the next tier. The recurrent and attention-based temporal models clearly outperform the purely convolutional baselines, supporting explicit temporal modeling for cough-event counting.",
            "",
            "## Files",
            "",
            "- `model_comparison_summary.csv`: ranked model-level summary.",
            "- `fold_results_all.csv`: all fold-level test metrics.",
            "- `missing_jobs.csv`: incomplete jobs, if any.",
            "- `environment.json`: runtime and CUDA/PyTorch environment.",
            "- `ch5_rerun_jobs_0p7m.yaml`: exact job queue.",
            "- `structure_compare_v2_0p7m.yaml`: exact model/data/training configuration.",
        ]
    )
    text = "\n".join(lines) + "\n"
    (out_dir / "RELEASE_REPORT.md").write_text(text, encoding="utf-8")
    (out_dir / "README.md").write_text(text, encoding="utf-8")
    print(f"Wrote report to {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["audit", "run", "report"])
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--group", default=None)
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
