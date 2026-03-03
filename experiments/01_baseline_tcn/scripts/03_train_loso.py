from __future__ import annotations

import copy
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm import tqdm

from coughcount.data.splits import build_loso_fold, load_subject_ids, write_subject_splits
from coughcount.evaluation.edgeai import evaluate_run_on_split
from coughcount.losses import count_mae, sample_count_abs_error, train_loss_weighted
from coughcount.paths import ProjectPaths as P
from coughcount.training.edgeai import (
    build_dynamic_pos_neg_loss_balancer,
    evaluate_counting_metrics,
    prepare_training_components,
    save_epoch_artifacts,
    save_run_config,
)
from coughcount.utils.io import atomic_write_json
from coughcount.utils.runtime import pick_device, set_seed


def load_config() -> dict:
    """从experiment.yaml加载配置"""
    exp_dir = Path(__file__).parent.parent
    config_path = exp_dir / "experiment.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def count_trainable_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _loss_balancer_state(
    loss_balancer: Any,
) -> dict[str, float | None]:
    return {
        "ema_pos_error": (
            float(loss_balancer.ema_pos_error)
            if loss_balancer.ema_pos_error is not None
            else None
        ),
        "ema_neg_error": (
            float(loss_balancer.ema_neg_error)
            if loss_balancer.ema_neg_error is not None
            else None
        ),
    }


def _restore_loss_balancer_state(loss_balancer: Any, state: dict[str, Any]) -> None:
    loss_balancer.ema_pos_error = (
        float(state["ema_pos_error"]) if state.get("ema_pos_error") is not None else None
    )
    loss_balancer.ema_neg_error = (
        float(state["ema_neg_error"]) if state.get("ema_neg_error") is not None else None
    )


def _infer_best_values(history: list[dict[str, Any]]) -> tuple[float, float]:
    if not history:
        return float("inf"), float("inf")
    best_val = float(min(float(h.get("val_mse", float("inf"))) for h in history))
    best_count = float(
        min(float(h.get("val_count_mae", float("inf"))) for h in history)
    )
    return best_val, best_count


def train_single_run(
    cfg: dict[str, Any],
    *,
    run_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(run_dir, cfg)

    train_cfg = cfg.get("train", {})
    device = pick_device(train_cfg.get("device", "cuda"))
    print(f"\n[train] run_dir={run_dir}")
    print(f"[train] device={device}")

    components = prepare_training_components(cfg, device=device)
    loss_balancer = build_dynamic_pos_neg_loss_balancer(cfg)

    print(
        f"[train] train_windows={len(components.train_dataset)} "
        f"(pos={len(components.train_dataset.pos_idx)} neg={len(components.train_dataset.neg_idx)})"
    )
    print(
        f"[train] val_windows={len(components.val_dataset)} "
        f"(pos={components.val_pos_windows} neg={components.val_neg_windows})"
        f"{' [balanced 1:1]' if components.val_is_balanced else ''}"
    )
    print(
        f"[train] model={cfg['model']['name']} "
        f"params={count_trainable_params(components.model):,}"
    )

    history_path = run_dir / "history.json"
    ckpt_last = run_dir / "last.pt"

    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_count = float("inf")

    if history_path.exists():
        loaded = _load_json(history_path)
        if isinstance(loaded, list):
            history = loaded
            best_val, best_count = _infer_best_values(history)

    if resume and ckpt_last.exists():
        ckpt = torch.load(ckpt_last, map_location="cpu")
        components.model.load_state_dict(ckpt["model_state"])
        components.optimizer.load_state_dict(ckpt["opt_state"])
        if ckpt.get("scheduler_state") is not None:
            components.scheduler.load_state_dict(ckpt["scheduler_state"])
        if ckpt.get("loss_balancer_state") is not None:
            _restore_loss_balancer_state(loss_balancer, ckpt["loss_balancer_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[train] resume from epoch {start_epoch}")

    for epoch in range(start_epoch, components.epochs + 1):
        components.model.train()
        train_losses: list[float] = []
        train_maes: list[float] = []

        pbar = tqdm(
            components.train_loader,
            desc=f"train e{epoch}/{components.epochs}",
            dynamic_ncols=True,
        )
        for batch in pbar:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            lengths = batch["lengths"].to(device)

            sample_weights, is_pos, pos_ratio = loss_balancer.build_sample_weights(
                y,
                lengths,
                epoch=epoch,
            )

            pred = components.model(x)
            loss = train_loss_weighted(
                pred,
                y,
                lengths,
                count_loss_weight=components.count_loss_weight,
                sample_weights=sample_weights if loss_balancer.enabled else None,
            )

            components.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            components.optimizer.step()

            train_losses.append(float(loss.item()))
            train_maes.append(float(count_mae(pred, y, lengths).item()))
            with np.errstate(invalid="ignore"):
                batch_count_errors = sample_count_abs_error(pred.detach(), y, lengths)
            loss_balancer.update_from_batch_errors(batch_count_errors, is_pos)

            epoch_progress = (epoch - 1) + (pbar.n / max(1, len(components.train_loader)))
            components.scheduler.step(epoch_progress)

            lr_now = float(components.optimizer.param_groups[0]["lr"])
            pbar.set_postfix(
                mse=f"{np.mean(train_losses):.4f}",
                cmae=f"{np.mean(train_maes):.3f}",
                lr=f"{lr_now:.2e}",
                pos_a=f"{pos_ratio:.2f}" if loss_balancer.enabled else "0.50",
            )

        train_mse = float(np.mean(train_losses)) if train_losses else float("nan")
        train_cmae = float(np.mean(train_maes)) if train_maes else float("nan")

        val_stats = evaluate_counting_metrics(
            components.model,
            components.val_loader,
            device,
            pos_threshold=components.pos_threshold,
            desc="val",
        )
        val_mse = float(val_stats["mse"])
        val_cmae = float(val_stats["count_mae"])
        lr_now = float(components.optimizer.param_groups[0]["lr"])

        rec = {
            "epoch": int(epoch),
            "lr": lr_now,
            "train_mse": train_mse,
            "train_count_mae": train_cmae,
            "val_mse": val_mse,
            "val_count_mae": val_cmae,
            "val_count_mae_pos": float(val_stats["count_mae_pos"]),
            "val_count_mae_neg": float(val_stats["count_mae_neg"]),
            "val_mean_pred_count_on_neg": float(val_stats["mean_pred_count_on_neg"]),
            "val_mean_gt_count_on_pos": float(val_stats["mean_gt_count_on_pos"]),
        }
        history.append(rec)

        print(
            f"[epoch {epoch}] "
            f"lr={lr_now:.2e} "
            f"train_mse={train_mse:.6f} train_count_mae={train_cmae:.4f} "
            f"val_mse={val_mse:.6f} val_count_mae={val_cmae:.4f} "
            f"(pos_mae={val_stats['count_mae_pos']:.3f} neg_mae={val_stats['count_mae_neg']:.3f})"
        )

        best_val, best_count = save_epoch_artifacts(
            run_dir=run_dir,
            cfg=cfg,
            epoch=epoch,
            model=components.model,
            optimizer=components.optimizer,
            scheduler=components.scheduler,
            val_mse=val_mse,
            val_count_mae=val_cmae,
            best_val_mse=best_val,
            best_val_count_mae=best_count,
            count_loss_weight=components.count_loss_weight,
            epoch_metrics=rec,
            history=history,
            extra_state={"loss_balancer_state": _loss_balancer_state(loss_balancer)},
        )

    out = {
        "best_val_mse": float(best_val),
        "best_val_count_mae": float(best_count),
        "epochs_total": int(components.epochs),
        "epochs_ran": int(max(0, components.epochs - start_epoch + 1)),
        "run_dir": str(run_dir),
    }
    atomic_write_json(run_dir / "train_summary.json", out)
    return out


def summarize_model_folds(model: str, fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    count_mae_values = [float(x["test_metrics"]["count_mae"]) for x in fold_results]
    pos_mae_values = [float(x["test_metrics"]["count_mae_pos"]) for x in fold_results]
    neg_mae_values = [float(x["test_metrics"]["count_mae_neg"]) for x in fold_results]

    def _summary(xs: list[float]) -> dict[str, float]:
        return {
            "mean": float(statistics.mean(xs)),
            "median": float(statistics.median(xs)),
            "std": float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0,
            "min": float(min(xs)),
            "max": float(max(xs)),
        }

    return {
        "model": model,
        "num_folds": int(len(fold_results)),
        "count_mae": _summary(count_mae_values),
        "count_mae_pos": _summary(pos_mae_values),
        "count_mae_neg": _summary(neg_mae_values),
    }


def prepare_cfg_for_run(
    base_cfg: dict[str, Any],
    *,
    model_name: str,
    splits_json: Path,
    fold_seed: int,
    epochs_override: int | None = None,
    device_override: str | None = None,
    batch_size_override: int | None = None,
    num_workers_override: int | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("data", {})
    cfg.setdefault("loader", {})
    cfg.setdefault("train", {})
    cfg.setdefault("model", {})

    cfg["seed"] = int(fold_seed)
    cfg["data"]["split_train"] = "train"
    cfg["data"]["split_val"] = "val"
    cfg["data"]["splits_json"] = str(Path(splits_json).resolve())
    cfg["model"]["name"] = str(model_name)

    if epochs_override is not None:
        cfg["train"]["epochs"] = int(epochs_override)
    if device_override is not None:
        cfg["train"]["device"] = str(device_override)
    if batch_size_override is not None:
        cfg["loader"]["batch_size"] = int(batch_size_override)
    if num_workers_override is not None:
        cfg["loader"]["num_workers"] = int(num_workers_override)

    return cfg


def run_fold(
    *,
    base_cfg: dict[str, Any],
    exp_root: Path,
    model_name: str,
    fold_index: int,
    test_subject: str,
    val_subject: str,
    split_json: Path,
    resume: bool,
    epochs_override: int | None = None,
    device_override: str | None = None,
    batch_size_override: int | None = None,
    num_workers_override: int | None = None,
) -> dict[str, Any]:
    fold_tag = f"fold_{fold_index:02d}_{test_subject}"
    run_dir = exp_root / model_name / fold_tag
    fold_result_path = run_dir / "fold_result.json"

    if resume and fold_result_path.exists():
        cached = _load_json(fold_result_path)
        if isinstance(cached, dict):
            print(f"[resume] skip completed {model_name}/{fold_tag}")
            return cached

    base_seed = int(base_cfg.get("seed", 0))
    fold_seed = int(base_seed + fold_index)
    cfg = prepare_cfg_for_run(
        base_cfg,
        model_name=model_name,
        splits_json=split_json,
        fold_seed=fold_seed,
        epochs_override=epochs_override,
        device_override=device_override,
        batch_size_override=batch_size_override,
        num_workers_override=num_workers_override,
    )

    set_seed(fold_seed)
    train_summary = train_single_run(cfg, run_dir=run_dir, resume=resume)

    eval_batch = int(cfg.get("loader", {}).get("batch_size", 16))
    eval_workers = int(cfg.get("loader", {}).get("num_workers", 4))
    eval_device = str(cfg.get("train", {}).get("device", "cuda"))
    test_metrics, out_file, ckpt_path = evaluate_run_on_split(
        run_dir,
        split="test",
        batch_size=eval_batch,
        num_workers=eval_workers,
        device_name=eval_device,
    )
    print(f"[test] model={model_name} fold={fold_index} ckpt={ckpt_path.name} out={out_file}")

    fold_result = {
        "model": model_name,
        "fold_index": int(fold_index),
        "test_subject": str(test_subject),
        "val_subject": str(val_subject),
        "run_dir": str(run_dir.resolve()),
        "checkpoint": str(ckpt_path.resolve()),
        "train_summary": train_summary,
        "test_metrics": test_metrics,
    }
    atomic_write_json(fold_result_path, fold_result)
    return fold_result


def train_final_best_model(
    *,
    base_cfg: dict[str, Any],
    exp_root: Path,
    best_model: str,
    subjects: list[str],
    resume: bool,
    epochs_override: int | None = None,
    device_override: str | None = None,
    batch_size_override: int | None = None,
    num_workers_override: int | None = None,
) -> dict[str, Any]:
    if len(subjects) < 2:
        raise ValueError("Final training requires at least 2 subjects.")

    val_subject = subjects[0]
    train_subjects = subjects[1:]
    split_json = exp_root / "splits" / "final_train_split.json"
    write_subject_splits(
        split_json,
        train=train_subjects,
        val=[val_subject],
        test=[],
        meta={
            "protocol": "final_train",
            "val_subject": str(val_subject),
            "n_train_subjects": int(len(train_subjects)),
        },
    )

    cfg = prepare_cfg_for_run(
        base_cfg,
        model_name=best_model,
        splits_json=split_json,
        fold_seed=int(base_cfg.get("seed", 0)),
        epochs_override=epochs_override,
        device_override=device_override,
        batch_size_override=batch_size_override,
        num_workers_override=num_workers_override,
    )
    set_seed(int(cfg.get("seed", 0)))

    run_dir = exp_root / best_model / "final_best"
    train_summary = train_single_run(cfg, run_dir=run_dir, resume=resume)
    final_result = {
        "model": best_model,
        "run_dir": str(run_dir.resolve()),
        "split_json": str(split_json.resolve()),
        "train_summary": train_summary,
        "best_checkpoint": str((run_dir / "best.pt").resolve()),
    }
    atomic_write_json(run_dir / "final_result.json", final_result)
    return final_result


def main() -> None:
    import argparse

    cfg = load_config()

    # Parse CLI args for LOSO-specific options, use config defaults
    parser = argparse.ArgumentParser(description="Run LOSO (Leave-One-Subject-Out) training.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=cfg.get("loso", {}).get("models", [cfg["model"]["name"]]),
        help="Models to train (default: from config or single model)",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=cfg.get("loso", {}).get("exp_name", ""),
        help="Experiment name (default: from config or auto-generated)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=cfg.get("loso", {}).get("epochs"),
        help="Epochs override (default: from config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=cfg.get("loso", {}).get("device"),
        help="Device override (default: from config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=cfg.get("loso", {}).get("batch_size"),
        help="Batch size override (default: from config)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=cfg.get("loso", {}).get("num_workers"),
        help="Num workers override (default: from config)",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=None,
        help="Number of folds (default: all subjects)",
    )
    parser.add_argument(
        "--start-fold",
        type=int,
        default=0,
        help="Start fold index (default: 0)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from existing checkpoints",
    )
    parser.add_argument(
        "--skip-final-train",
        action="store_true",
        help="Skip final best model training",
    )
    args = parser.parse_args()

    model_names = [str(m).lower() for m in args.models]
    resume = not bool(args.no_resume)

    data_cfg = cfg.get("data", {})
    manifest_csv = Path(str(data_cfg.get("splits_json", P.edgeai_manifest_csv)).replace("splits.json", "manifest.csv"))
    if not manifest_csv.exists():
        manifest_csv = P.edgeai_manifest_csv
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")

    subjects = load_subject_ids(manifest_csv)
    if len(subjects) < 3:
        raise RuntimeError(f"Need >=3 subjects for LOSO. found={len(subjects)}")

    num_folds = len(subjects) if args.num_folds is None else int(args.num_folds)
    num_folds = max(1, min(num_folds, len(subjects)))
    start_fold = int(args.start_fold)
    if start_fold < 0 or start_fold >= len(subjects):
        raise ValueError(f"start_fold out of range: {start_fold}")

    exp_name = args.exp_name.strip() if args.exp_name.strip() else ""
    if not exp_name:
        exp_name = f"edgeai_loso_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_root = P.runs / exp_name
    (exp_root / "splits").mkdir(parents=True, exist_ok=True)

    print(f"Experiment root: {exp_root}")
    print(f"Subjects: {len(subjects)}")
    print(f"Models: {model_names}")
    print(f"Folds: start={start_fold} num={num_folds}")
    print(f"Resume: {resume}")

    all_results: dict[str, list[dict[str, Any]]] = {m: [] for m in model_names}

    for k in range(num_folds):
        fold_index = (start_fold + k) % len(subjects)
        fold = build_loso_fold(subjects, fold_index=fold_index)
        test_subject = str(fold["test"][0])
        val_subject = str(fold["val"][0])
        split_json = exp_root / "splits" / f"fold_{fold_index:02d}_{test_subject}.json"
        write_subject_splits(
            split_json,
            train=list(fold["train"]),
            val=list(fold["val"]),
            test=list(fold["test"]),
            meta={
                "protocol": "loso_with_subject_val",
                "fold_index": int(fold_index),
                "test_subject": test_subject,
                "val_subject": val_subject,
            },
        )
        print(
            f"\n=== Fold {fold_index:02d} | test={test_subject} | val={val_subject} ==="
        )

        for model_name in model_names:
            res = run_fold(
                base_cfg=cfg,
                exp_root=exp_root,
                model_name=model_name,
                fold_index=fold_index,
                test_subject=test_subject,
                val_subject=val_subject,
                split_json=split_json,
                resume=resume,
                epochs_override=args.epochs,
                device_override=args.device,
                batch_size_override=args.batch_size,
                num_workers_override=args.num_workers,
            )
            all_results[model_name].append(res)

        atomic_write_json(exp_root / "fold_results.json", all_results)

    model_summaries: list[dict[str, Any]] = []
    for model_name in model_names:
        model_summary = summarize_model_folds(model_name, all_results[model_name])
        model_summaries.append(model_summary)
        print(
            f"[summary] {model_name}: "
            f"count_mae(mean={model_summary['count_mae']['mean']:.4f}, "
            f"median={model_summary['count_mae']['median']:.4f}, "
            f"std={model_summary['count_mae']['std']:.4f})"
        )

    best_model = min(model_summaries, key=lambda x: x["count_mae"]["mean"])["model"]
    print(f"[summary] best_model_by_mean_count_mae={best_model}")

    final_result: dict[str, Any] | None = None
    if not args.skip_final_train:
        final_result = train_final_best_model(
            base_cfg=cfg,
            exp_root=exp_root,
            best_model=best_model,
            subjects=subjects,
            resume=resume,
            epochs_override=args.epochs,
            device_override=args.device,
            batch_size_override=args.batch_size,
            num_workers_override=args.num_workers,
        )
        print(f"[final] best checkpoint: {final_result['best_checkpoint']}")

    payload = {
        "experiment_root": str(exp_root.resolve()),
        "models": model_names,
        "num_subjects": int(len(subjects)),
        "num_folds": int(num_folds),
        "start_fold": int(start_fold),
        "summaries": model_summaries,
        "winner": best_model,
        "final_result": final_result,
    }
    atomic_write_json(exp_root / "summary.json", payload)
    print(f"Done. summary={exp_root / 'summary.json'}")


if __name__ == "__main__":
    main()
