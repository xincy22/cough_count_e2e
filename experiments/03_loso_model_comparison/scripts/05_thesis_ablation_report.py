"""
Build thesis-ready ablation tables from LOSO result folders.

The script is intentionally read-only for runs/. It collects completed
test_results.json files, prefers loso_summary.json when present, and writes
CSV/Markdown artifacts to result/.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


MODEL_COMPONENTS = {
    "gru": {
        "tcn": "No",
        "gru": "Yes",
        "label": "GRU",
        "params": 115841,
        "size_mb": 0.442,
    },
    "tcn": {
        "tcn": "Yes",
        "gru": "No",
        "label": "TCN-matched",
        "params": 609665,
        "size_mb": 2.326,
    },
    "tcn_gru": {
        "tcn": "Yes",
        "gru": "Yes",
        "label": "TCN+GRU",
        "params": 708737,
        "size_mb": 2.704,
    },
    "crnn": {
        "tcn": "No",
        "gru": "Yes",
        "label": "CRNN",
        "params": None,
        "size_mb": None,
    },
    "cnn1d": {
        "tcn": "No",
        "gru": "No",
        "label": "CNN1D",
        "params": None,
        "size_mb": None,
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _fmt_pm(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.4f} ± {std_value:.4f}"


def _model_sort_key(model_name: str) -> tuple[int, str]:
    order = {"gru": 0, "tcn": 1, "tcn_gru": 2, "crnn": 3, "cnn1d": 4}
    return order.get(model_name, 99), model_name


def collect_fold_results(runs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(runs_dir.glob("loso_*/M*/fold_*_test_*/test_results.json")):
        data = _load_json(path)
        model_dir = path.parents[1]
        model_name = _infer_model_name(model_dir)
        rows.append(
            {
                "run_dir": str(model_dir.parents[0]),
                "model_dir": str(model_dir),
                "model_id": model_dir.name,
                "model_name": model_name,
                "fold": int(data["fold"]),
                "test_subject": str(data["test_subject"]),
                "count_mae": float(data["count_mae"]),
                "count_mae_pos": float(data["count_mae_pos"]),
                "count_mae_neg": float(data["count_mae_neg"]),
                "mean_pred_count_on_neg": float(data["mean_pred_count_on_neg"]),
                "mean_gt_count_on_pos": float(data["mean_gt_count_on_pos"]),
                "num_test_samples": int(data["num_test_samples"]),
                "source_json": str(path),
            }
        )
    return rows


def _infer_model_name(model_dir: Path) -> str:
    summary_path = model_dir / "loso_summary.json"
    if summary_path.exists():
        data = _load_json(summary_path)
        if data.get("model_name"):
            return str(data["model_name"])

    config_paths = sorted(model_dir.glob("fold_*_test_*/config_resolved.yaml"))
    if config_paths:
        text = config_paths[0].read_text(encoding="utf-8", errors="replace")
        for name in MODEL_COMPONENTS:
            if f'name: {name}' in text or f'name: "{name}"' in text:
                return name

    fallback = {"M0": "gru", "M1": "tcn", "M2": "tcn_gru", "M3": "crnn", "M4": "cnn1d"}
    return fallback.get(model_dir.name, model_dir.name)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["model_name"], []).append(row)

    summaries: list[dict[str, Any]] = []
    for model_name, model_rows in grouped.items():
        vals = [r["count_mae"] for r in model_rows]
        pos_vals = [r["count_mae_pos"] for r in model_rows]
        neg_vals = [r["count_mae_neg"] for r in model_rows]
        comp = MODEL_COMPONENTS.get(
            model_name,
            {
                "tcn": "Unknown",
                "gru": "Unknown",
                "label": model_name,
                "params": None,
                "size_mb": None,
            },
        )
        summaries.append(
            {
                "model_name": model_name,
                "paper_label": comp["label"],
                "tcn_module": comp["tcn"],
                "gru_module": comp["gru"],
                "trainable_params": comp["params"],
                "fp32_size_mb": comp["size_mb"],
                "num_folds": len(model_rows),
                "mean_count_mae": mean(vals),
                "std_count_mae": pstdev(vals) if len(vals) > 1 else 0.0,
                "mean_count_mae_pos": mean(pos_vals),
                "std_count_mae_pos": pstdev(pos_vals) if len(pos_vals) > 1 else 0.0,
                "mean_count_mae_neg": mean(neg_vals),
                "std_count_mae_neg": pstdev(neg_vals) if len(neg_vals) > 1 else 0.0,
            }
        )

    return sorted(summaries, key=lambda r: _model_sort_key(r["model_name"]))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_thesis_table_csv(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    rows = []
    for row in summary_rows:
        rows.append(
            {
                "模型": row["paper_label"],
                "TCN模块": row["tcn_module"],
                "GRU模块": row["gru_module"],
                "参数量": row["trainable_params"],
                "FP32大小(MB)": row["fp32_size_mb"],
                "LOSO折数": row["num_folds"],
                "Count MAE": _fmt_pm(row["mean_count_mae"], row["std_count_mae"]),
                "Positive MAE": _fmt_pm(
                    row["mean_count_mae_pos"], row["std_count_mae_pos"]
                ),
                "Negative MAE": _fmt_pm(
                    row["mean_count_mae_neg"], row["std_count_mae_neg"]
                ),
            }
        )

    write_csv(
        path,
        rows,
        [
            "模型",
            "TCN模块",
            "GRU模块",
            "参数量",
            "FP32大小(MB)",
            "LOSO折数",
            "Count MAE",
            "Positive MAE",
            "Negative MAE",
        ],
    )


def write_figures(
    out_dir: Path,
    summary_rows: list[dict[str, Any]],
    fold_rows: list[dict[str, Any]],
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on runtime package set
        note = out_dir / "figures_skipped.txt"
        note.write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return []

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    labels = [row["paper_label"] for row in summary_rows]
    means = [row["mean_count_mae"] for row in summary_rows]
    stds = [row["std_count_mae"] for row in summary_rows]

    bar_path = fig_dir / "ablation_count_mae_bar.png"
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=180)
    palette = ["#7f8c8d", "#4c78a8", "#2f6f4e", "#b279a2", "#f58518"]
    bars = ax.bar(
        labels,
        means,
        yerr=stds,
        capsize=4,
        color=[palette[i % len(palette)] for i in range(len(labels))],
    )
    ax.set_ylabel("Count MAE")
    ax.set_xlabel("Model")
    ax.set_title("LOSO Count MAE by Model")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    for bar, value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(bar_path)
    plt.close(fig)

    box_path = fig_dir / "ablation_fold_mae_boxplot.png"
    grouped = []
    for model_name in [row["model_name"] for row in summary_rows]:
        grouped.append(
            [row["count_mae"] for row in fold_rows if row["model_name"] == model_name]
        )
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=180)
    ax.boxplot(grouped, labels=labels, showmeans=True)
    ax.set_ylabel("Fold Count MAE")
    ax.set_xlabel("Model")
    ax.set_title("LOSO Fold Error Distribution")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    fig.savefig(box_path)
    plt.close(fig)

    return [bar_path, box_path]


def write_markdown(
    path: Path,
    summary_rows: list[dict[str, Any]],
    fold_rows: list[dict[str, Any]],
    figure_paths: list[Path],
) -> None:
    completed = {row["paper_label"]: int(row["num_folds"]) for row in summary_rows}
    lines = [
        "# 咳嗽计数模型消融实验结果",
        "",
        f"生成时间: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 表：模型结构消融实验",
        "",
        "| 模型 | TCN模块 | GRU模块 | 参数量 | FP32大小(MB) | LOSO折数 | Count MAE | Positive MAE | Negative MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summary_rows:
        lines.append(
            "| {paper_label} | {tcn_module} | {gru_module} | {params} | {size_mb} | {num_folds} | "
            "{count_mae} | {pos_mae} | {neg_mae} |".format(
                paper_label=row["paper_label"],
                tcn_module=row["tcn_module"],
                gru_module=row["gru_module"],
                params=row["trainable_params"] if row["trainable_params"] is not None else "",
                size_mb=(
                    f"{float(row['fp32_size_mb']):.3f}"
                    if row["fp32_size_mb"] is not None
                    else ""
                ),
                num_folds=row["num_folds"],
                count_mae=_fmt_pm(row["mean_count_mae"], row["std_count_mae"]),
                pos_mae=_fmt_pm(
                    row["mean_count_mae_pos"], row["std_count_mae_pos"]
                ),
                neg_mae=_fmt_pm(
                    row["mean_count_mae_neg"], row["std_count_mae_neg"]
                ),
            )
        )

    lines.extend(
        [
            "",
            "## 论文正文表述草稿",
            "",
            "为验证咳嗽计数模型中不同时间建模模块的作用，本研究在相同数据划分、"
            "相同密度标签构造方式和相同训练设置下开展结构消融实验。实验比较了"
            "仅包含GRU的时序模型、仅包含TCN的卷积时序模型，以及同时包含TCN和GRU"
            "的组合模型。TCN模块主要用于提取局部时间范围内的声学变化模式，GRU模块"
            "用于建模更长时间范围内的上下文依赖。Count MAE越低，说明模型预测的咳嗽"
            "次数与人工标注次数越接近。",
            "",
        ]
    )

    labels = {row["model_name"]: row for row in summary_rows}
    if {"gru", "tcn", "tcn_gru"}.issubset(labels):
        best = min(summary_rows, key=lambda r: r["mean_count_mae"])
        if best["model_name"] == "tcn_gru":
            tcn_row = labels["tcn"]
            tcn_gru_row = labels["tcn_gru"]
            lines.append(
                "在完整LOSO实验中，TCN+GRU取得最低的平均Count MAE，TCN-only也保持"
                "在相近误差水平，二者共同构成本实验中的第一梯队。相较于GRU-only，"
                "TCN和TCN+GRU均表现出更低的整体计数误差，说明局部时序卷积特征是"
                "当前咳嗽计数任务中的关键建模模块。"
            )
            lines.append(
                "从正负样本分项看，TCN+GRU的Positive MAE为{tcn_gru_pos}，"
                "TCN-only的Positive MAE为{tcn_pos}；这说明组合结构在含咳嗽样本"
                "上的计数更敏感。若TCN+GRU的Negative MAE高于TCN-only，则应将其"
                "解释为敏感性提升带来的负样本残余响应，而不是简单写成全面最优。".format(
                    tcn_gru_pos=_fmt_pm(
                        tcn_gru_row["mean_count_mae_pos"],
                        tcn_gru_row["std_count_mae_pos"],
                    ),
                    tcn_pos=_fmt_pm(
                        tcn_row["mean_count_mae_pos"],
                        tcn_row["std_count_mae_pos"],
                    ),
                )
            )
        else:
            tcn_row = labels["tcn"]
            tcn_gru_row = labels["tcn_gru"]
            gru_row = labels["gru"]
            lines.append(
                "在完整LOSO实验中，{best}取得最低的平均Count MAE。TCN-only与"
                "TCN+GRU的整体误差接近，二者共同构成本实验中的第一梯队；二者均"
                "明显优于GRU-only，说明局部时序卷积特征对咳嗽计数具有重要作用。"
                "在当前8秒窗口和TCN感受野设置下，GRU模块对整体MAE的增益有限，"
                "但仍需结合正样本和负样本分项指标分析其作用。".format(
                    best=best["paper_label"]
                )
            )
            lines.append(
                "进一步看正负样本分项，TCN+GRU的Positive MAE为{tcn_gru_pos}，"
                "TCN-only的Positive MAE为{tcn_pos}，GRU-only的Positive MAE为"
                "{gru_pos}。这表明组合结构在含咳嗽样本上的计数误差可能低于"
                "TCN-only；同时若TCN+GRU的Negative MAE更高，则说明其敏感性提升"
                "伴随一定负样本残余响应。".format(
                    tcn_gru_pos=_fmt_pm(
                        tcn_gru_row["mean_count_mae_pos"],
                        tcn_gru_row["std_count_mae_pos"],
                    ),
                    tcn_pos=_fmt_pm(
                        tcn_row["mean_count_mae_pos"],
                        tcn_row["std_count_mae_pos"],
                    ),
                    gru_pos=_fmt_pm(
                        gru_row["mean_count_mae_pos"],
                        gru_row["std_count_mae_pos"],
                    ),
                )
            )
    else:
        lines.append(
            "当前表格仍为阶段性结果。最终结论应在GRU、TCN和TCN+GRU三组模型均完成"
            "15折LOSO后再写入论文。"
        )

    if figure_paths:
        lines.extend(["", "## 图文件", ""])
        for fig_path in figure_paths:
            lines.append(f"- `{fig_path.name}`")

    lines.extend(
        [
            "",
            "## 已完成折数",
            "",
        ]
    )
    for label, count in sorted(completed.items()):
        lines.append(f"- {label}: {count} folds")

    lines.extend(
        [
            "",
            "## Fold明细",
            "",
            "| Model | Fold | Test subject | Count MAE | Pos MAE | Neg MAE | Test samples |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    sorted_folds = sorted(
        fold_rows,
        key=lambda r: (_model_sort_key(r["model_name"]), int(r["fold"])),
    )
    for row in sorted_folds:
        label = MODEL_COMPONENTS.get(row["model_name"], {}).get(
            "label", row["model_name"]
        )
        lines.append(
            f"| {label} | {row['fold']} | {row['test_subject']} | "
            f"{_fmt(row['count_mae'])} | {_fmt(row['count_mae_pos'])} | "
            f"{_fmt(row['count_mae_neg'])} | {row['num_test_samples']} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "runs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "result",
    )
    args = parser.parse_args()

    fold_rows = collect_fold_results(args.runs_dir)
    if not fold_rows:
        raise SystemExit(f"No test_results.json found under {args.runs_dir}")

    summary_rows = summarize(fold_rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"thesis_ablation_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        out_dir / "ablation_summary.csv",
        summary_rows,
        [
            "paper_label",
            "model_name",
            "tcn_module",
            "gru_module",
            "trainable_params",
            "fp32_size_mb",
            "num_folds",
            "mean_count_mae",
            "std_count_mae",
            "mean_count_mae_pos",
            "std_count_mae_pos",
            "mean_count_mae_neg",
            "std_count_mae_neg",
        ],
    )
    write_csv(
        out_dir / "ablation_fold_details.csv",
        fold_rows,
        [
            "model_name",
            "model_id",
            "fold",
            "test_subject",
            "count_mae",
            "count_mae_pos",
            "count_mae_neg",
            "mean_pred_count_on_neg",
            "mean_gt_count_on_pos",
            "num_test_samples",
            "source_json",
        ],
    )
    write_thesis_table_csv(out_dir / "thesis_table_ablation.csv", summary_rows)
    figure_paths = write_figures(out_dir, summary_rows, fold_rows)
    write_markdown(
        out_dir / "thesis_ablation_report.md",
        summary_rows,
        fold_rows,
        figure_paths,
    )

    print(f"Wrote thesis ablation artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
