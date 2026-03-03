from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

from coughcount.data.splits import make_loso_subject_splits
from coughcount.evaluation.edgeai import evaluate_run_on_split
from coughcount.paths import ProjectPaths as P
from coughcount.runtime import atomic_write_json
from coughcount.training.edgeai import train_edgeai


def _load_loso_index(index_path: Path) -> list[dict[str, Any]]:
    obj = json.loads(index_path.read_text(encoding="utf-8"))
    folds = obj.get("folds", [])
    if not isinstance(folds, list):
        raise ValueError(f"Invalid LOSO index format: {index_path}")
    out: list[dict[str, Any]] = []
    for x in folds:
        if isinstance(x, dict):
            out.append(dict(x))
    return out


def _load_existing_summary(summary_path: Path) -> list[dict[str, Any]]:
    if not summary_path.exists():
        return []

    try:
        obj = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if not isinstance(obj, list):
        return []

    out: list[dict[str, Any]] = []
    for x in obj:
        if isinstance(x, dict):
            out.append(dict(x))
    return out


def prepare_loso_folds(
    *,
    manifest_csv: Path,
    out_dir: Path,
    val_subjects: int,
    seed: int,
    regenerate: bool = False,
) -> list[dict[str, Any]]:
    index_path = Path(out_dir) / "index.json"
    if index_path.exists() and (not regenerate):
        return _load_loso_index(index_path)
    return make_loso_subject_splits(
        manifest_csv,
        out_dir,
        val_subjects=val_subjects,
        seed=seed,
    )


def _fold_cfg(base_cfg: dict[str, Any], splits_json: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    data_cfg = dict(cfg.get("data", {}))
    data_cfg["splits_json"] = str(Path(splits_json))
    data_cfg["split_train"] = "train"
    data_cfg["split_val"] = "val"
    cfg["data"] = data_cfg
    return cfg


def run_loso_training(
    base_cfg: dict[str, Any],
    *,
    manifest_csv: Path = P.edgeai_manifest_csv,
    splits_dir: Path = P.edgeai_processed / "loso",
    val_subjects: int = 1,
    split_seed: int = 42,
    regenerate_splits: bool = False,
    run_root: Path | None = None,
    start_fold: int = 0,
    max_folds: int | None = None,
    test_batch_size: int = 32,
    test_num_workers: int = 0,
    test_device: str = "cuda",
) -> tuple[Path, list[dict[str, Any]]]:
    folds = prepare_loso_folds(
        manifest_csv=manifest_csv,
        out_dir=splits_dir,
        val_subjects=val_subjects,
        seed=split_seed,
        regenerate=regenerate_splits,
    )
    if not folds:
        raise RuntimeError("No LOSO folds generated.")

    ts = time.strftime("%Y%m%d_%H%M%S")
    if run_root is None:
        run_root = P.runs / f"loso_{ts}"
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    summary_path = run_root / "loso_summary.json"
    summary = _load_existing_summary(summary_path)
    summary_by_fold: dict[int, dict[str, Any]] = {}
    for rec in summary:
        if not isinstance(rec, dict):
            continue
        try:
            summary_by_fold[int(rec.get("fold_index"))] = rec
        except Exception:
            continue

    fold_end = len(folds)
    if max_folds is not None:
        fold_end = min(fold_end, int(start_fold) + int(max_folds))

    for fold in folds[int(start_fold) : fold_end]:
        fold_idx = int(fold["fold_index"])
        test_subject = str(fold["test_subject"])
        splits_json = Path(str(fold["splits_json"]))
        print(f"\n===== LOSO fold {fold_idx} | test_subject={test_subject} =====")

        cfg_fold = _fold_cfg(base_cfg, splits_json)
        fold_run_dir = run_root / f"fold_{fold_idx:02d}_test_{test_subject}"
        test_results_path = fold_run_dir / "test_results_test.json"
        if test_results_path.exists() and fold_idx in summary_by_fold:
            print(f"Skip: existing fold results found: {test_results_path}")
            continue

        if test_results_path.exists():
            test_metrics = json.loads(test_results_path.read_text(encoding="utf-8"))
            out_file = test_results_path
        else:
            train_edgeai(cfg_fold, run_dir=fold_run_dir)
            test_metrics, out_file, _ = evaluate_run_on_split(
                fold_run_dir,
                split="test",
                batch_size=test_batch_size,
                num_workers=test_num_workers,
                device_name=test_device,
            )

        best_info_path = fold_run_dir / "best_info.json"
        best_info = {}
        if best_info_path.exists():
            best_info = json.loads(best_info_path.read_text(encoding="utf-8"))

        rec: dict[str, Any] = {
            "fold_index": fold_idx,
            "test_subject": test_subject,
            "splits_json": str(splits_json),
            "run_dir": str(fold_run_dir),
            "best_info": best_info,
            "test_metrics": test_metrics,
            "test_results_path": str(out_file),
        }
        summary_by_fold[fold_idx] = rec
        summary = [summary_by_fold[i] for i in sorted(summary_by_fold.keys())]
        atomic_write_json(summary_path, summary)

    summary = [summary_by_fold[i] for i in sorted(summary_by_fold.keys())]
    atomic_write_json(summary_path, summary)

    test_maes: list[float] = []
    pos_maes: list[float] = []
    for x in summary:
        tm = x.get("test_metrics", {})
        if not isinstance(tm, dict):
            continue
        if "count_mae" not in tm or "count_mae_pos" not in tm:
            continue
        test_maes.append(float(tm["count_mae"]))
        pos_maes.append(float(tm["count_mae_pos"]))

    if test_maes:
        mean_test_count_mae: float | None = float(sum(test_maes) / len(test_maes))
        mean_test_pos_mae: float | None = float(sum(pos_maes) / len(pos_maes))
    else:
        mean_test_count_mae = None
        mean_test_pos_mae = None

    agg = {
        "n_folds": int(len(test_maes)),
        "mean_test_count_mae": mean_test_count_mae,
        "mean_test_pos_mae": mean_test_pos_mae,
    }
    atomic_write_json(run_root / "loso_aggregate.json", agg)
    return run_root, summary
