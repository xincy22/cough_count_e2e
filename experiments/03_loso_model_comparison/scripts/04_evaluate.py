"""
结果整理脚本 - 整理LOSO实验结果到result文件夹
从runs目录读取结果，生成汇总报告并保存到result/
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


def load_config() -> dict:
    """从experiment.yaml加载配置"""
    exp_dir = Path(__file__).parent.parent
    config_path = exp_dir / "experiment.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def _load_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_loso_run(runs_dir: Path) -> Path:
    """找到最新的LOSO运行目录"""
    loso_dirs = sorted(runs_dir.glob("loso_*"), key=lambda p: p.name)
    return loso_dirs[-1] if loso_dirs else None


def summarize_loso_results(loso_run_dir: Path, cfg: dict) -> dict:
    """汇总LOSO结果"""

    summary_file = loso_run_dir / "loso_comparison_summary.json"
    if not summary_file.exists():
        print(f"Error: LOSO summary not found at {summary_file}")
        return None

    summary = _load_json(summary_file)

    # Extract key metrics
    results = []
    for model_result in summary.get("results", []):
        results.append({
            "model_id": model_result["model_id"],
            "model_name": model_result["model_name"],
            "mean_count_mae": model_result["mean_count_mae"],
            "std_count_mae": model_result["std_count_mae"],
            "mean_count_mae_pos": model_result["mean_count_mae_pos"],
            "std_count_mae_pos": model_result["std_count_mae_pos"],
            "mean_count_mae_neg": model_result["mean_count_mae_neg"],
            "std_count_mae_neg": model_result["std_count_mae_neg"],
            "num_folds": model_result["num_folds"],
        })

    # Find best model
    best_model = min(results, key=lambda x: x["mean_count_mae"])

    return {
        "experiment": cfg["name"],
        "timestamp": summary.get("timestamp"),
        "device": summary.get("device"),
        "density_kernel": cfg.get("density", {}),
        "results": results,
        "best_model": {
            "model_id": best_model["model_id"],
            "model_name": best_model["model_name"],
            "mean_count_mae": best_model["mean_count_mae"],
        },
        "summary": {
            "num_models": len(results),
            "num_folds_per_model": results[0]["num_folds"] if results else 0,
        }
    }


def save_results_to_result_folder(result: dict, exp_dir: Path) -> Path:
    """保存结果到result文件夹"""

    result_dir = exp_dir / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = result_dir / f"loso_summary_{timestamp}.json"

    with result_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result_file


def generate_markdown_report(result: dict, exp_dir: Path) -> Path:
    """生成Markdown格式的报告"""

    result_dir = exp_dir / "result"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = result_dir / f"loso_report_{timestamp}.md"

    lines = [
        "# LOSO Model Comparison Results",
        "",
        f"**Experiment**: {result['experiment']}",
        f"**Timestamp**: {result['timestamp']}",
        f"**Device**: {result['device']}",
        "",
        "## Density Kernel Configuration",
        "",
    ]

    kernel = result.get("density_kernel", {})
    kernel_type = kernel.get("kernel", "unknown")
    lines.append(f"- **Kernel**: {kernel_type}")

    if kernel_type == "gaussian":
        lines.append(f"- **sigma_sec**: {kernel.get('sigma_sec', 'N/A')}")
    elif kernel_type == "skewed_gaussian":
        lines.append(f"- **sigma_left_sec**: {kernel.get('sigma_left_sec', 'N/A')}")
        lines.append(f"- **sigma_right_sec**: {kernel.get('sigma_right_sec', 'N/A')}")
    elif kernel_type == "cosine":
        lines.append(f"- **half_width_sec**: {kernel.get('sigma_sec', 'N/A')}")

    lines.extend([
        "",
        "## Model Comparison",
        "",
        "| Model | Mean MAE | Std MAE | Mean Pos MAE | Std Pos MAE | Mean Neg MAE | Std Neg MAE |",
        "|-------|----------|---------|--------------|-------------|--------------|-------------|",
    ])

    for r in result["results"]:
        lines.append(
            f"| {r['model_name']} | "
            f"{r['mean_count_mae']:.4f} | "
            f"{r['std_count_mae']:.4f} | "
            f"{r['mean_count_mae_pos']:.4f} | "
            f"{r['std_count_mae_pos']:.4f} | "
            f"{r['mean_count_mae_neg']:.4f} | "
            f"{r['std_count_mae_neg']:.4f} |"
        )

    lines.extend([
        "",
        "## Best Model",
        "",
        f"- **Model**: {result['best_model']['model_name']}",
        f"- **Mean Count MAE**: {result['best_model']['mean_count_mae']:.4f}",
        "",
        "## Recommendation",
        "",
        f"Based on LOSO evaluation, **{result['best_model']['model_name']}** achieves the lowest "
        f"mean count MAE of {result['best_model']['mean_count_mae']:.4f} across {result['summary']['num_folds_per_model']} subjects.",
        "",
        "This model is recommended for the final training in experiment 04.",
    ])

    with report_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_file


def main() -> None:
    cfg = load_config()
    exp_dir = Path(__file__).parent.parent

    runs_dir = exp_dir / "runs"

    # Find latest LOSO run
    loso_run_dir = find_latest_loso_run(runs_dir)
    if not loso_run_dir:
        print("Error: No LOSO run found in runs/ directory")
        print("Please run scripts/03_loso.py first")
        return

    print(f"Using LOSO run: {loso_run_dir}")

    # Summarize results
    result = summarize_loso_results(loso_run_dir, cfg)
    if not result:
        return

    # Save to result folder
    result_file = save_results_to_result_folder(result, exp_dir)
    print(f"Results saved to: {result_file}")

    # Generate markdown report
    report_file = generate_markdown_report(result, exp_dir)
    print(f"Report saved to: {report_file}")

    # Print summary
    print(f"\n{'='*60}")
    print("LOSO Model Comparison Summary")
    print(f"{'='*60}")
    print(f"Best Model: {result['best_model']['model_name']}")
    print(f"Mean Count MAE: {result['best_model']['mean_count_mae']:.4f}")
    print()


if __name__ == "__main__":
    main()
