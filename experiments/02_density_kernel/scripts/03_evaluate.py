"""
结果整理脚本 - 整理density kernel对比实验结果到result文件夹
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


def collect_kernel_results(runs_dir: Path, cfg: dict) -> list[dict]:
    """收集所有kernel的训练和测试结果"""

    kernels = cfg["kernels"]
    results = []

    for kernel_cfg in kernels:
        kernel_id = kernel_cfg["id"]
        run_dir = runs_dir / kernel_id

        # Load train summary
        train_summary_path = run_dir / "train_summary.json"
        if not train_summary_path.exists():
            continue

        train_summary = _load_json(train_summary_path)

        # Load test results
        test_results_path = run_dir / "test_results_test.json"
        if not test_results_path.exists():
            continue

        test_results = _load_json(test_results_path)

        results.append({
            "kernel_id": kernel_id,
            "kernel_name": kernel_cfg["name"],
            "kernel_type": kernel_cfg["kernel"],
            "description": kernel_cfg["description"],
            "sigma_left_sec": kernel_cfg.get("sigma_left_sec"),
            "sigma_right_sec": kernel_cfg.get("sigma_right_sec"),
            "sigma_sec": kernel_cfg.get("sigma_sec"),
            "train_summary": train_summary,
            "test_metrics": test_results,
        })

    return results


def find_best_kernel(results: list[dict]) -> dict:
    """找出最佳kernel"""
    if not results:
        return None

    # 按test_count_mae排序
    best = min(results, key=lambda x: x["test_metrics"]["count_mae"])

    return {
        "kernel_id": best["kernel_id"],
        "kernel_name": best["kernel_name"],
        "kernel_type": best["kernel_type"],
        "sigma_left_sec": best["sigma_left_sec"],
        "sigma_right_sec": best["sigma_right_sec"],
        "sigma_sec": best["sigma_sec"],
        "test_count_mae": best["test_metrics"]["count_mae"],
        "test_count_mae_pos": best["test_metrics"]["count_mae_pos"],
    }


def save_results_to_result_folder(result: dict, exp_dir: Path) -> Path:
    """保存结果到result文件夹"""

    result_dir = exp_dir / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = result_dir / f"kernel_comparison_summary_{timestamp}.json"

    with result_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result_file


def generate_markdown_report(result: dict, exp_dir: Path) -> Path:
    """生成Markdown格式的报告"""

    result_dir = exp_dir / "result"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = result_dir / f"kernel_comparison_report_{timestamp}.md"

    lines = [
        "# Density Kernel Comparison Results",
        "",
        f"**Experiment**: {result['experiment']}",
        f"**Timestamp**: {result['timestamp']}",
        f"**Kernels Tested**: {result['kernels_tested']}",
        "",
        "## Summary",
        "",
        f"**Best Kernel**: {result['best_kernel']['kernel_name']} ({result['best_kernel']['kernel_id']})",
        "",
        "### Best Kernel Configuration",
        "",
    ]

    best = result['best_kernel']
    kernel_type = best['kernel_type']
    lines.append(f"- **Kernel Type**: {kernel_type}")

    if kernel_type == "gaussian":
        lines.append(f"- **sigma_sec**: {best['sigma_sec']}")
    elif kernel_type == "skewed_gaussian":
        lines.append(f"- **sigma_left_sec**: {best['sigma_left_sec']}")
        lines.append(f"- **sigma_right_sec**: {best['sigma_right_sec']}")
    elif kernel_type == "cosine":
        lines.append(f"- **half_width_sec**: {best['sigma_sec']}")

    lines.extend([
        "",
        "### Best Performance",
        "",
        f"- **Test Count MAE**: {best['test_count_mae']:.4f}",
        f"- **Test Count MAE (pos)**: {best['test_count_mae_pos']:.4f}",
        "",
        "## All Results",
        "",
        "| ID | Name | Type | sigma_left | sigma_right | Test MAE | Test Pos MAE | Test Neg MAE |",
        "|----|------|------|------------|-------------|----------|--------------|--------------|",
    ])

    for r in result["results"]:
        sigma_left = r.get("sigma_left_sec")
        sigma_right = r.get("sigma_right_sec")
        sigma = r.get("sigma_sec")

        if r["kernel_type"] == "gaussian":
            sigma_str = f"σ={sigma}"
        elif r["kernel_type"] == "skewed_gaussian":
            sigma_str = f"L={sigma_left}/R={sigma_right}"
        elif r["kernel_type"] == "cosine":
            sigma_str = f"hw={sigma}"
        else:
            sigma_str = "N/A"

        lines.append(
            f"| {r['kernel_id']} | "
            f"{r['kernel_name']} | "
            f"{r['kernel_type']} | "
            f"{sigma_left or '-'} | "
            f"{sigma_right or '-'} | "
            f"{r['test_metrics']['count_mae']:.4f} | "
            f"{r['test_metrics']['count_mae_pos']:.4f} | "
            f"{r['test_metrics']['count_mae_neg']:.4f} |"
        )

    lines.extend([
        "",
        "## Recommendation for Experiment 03",
        "",
        f"Based on the results, use **{best['kernel_name']}** for the LOSO model comparison in experiment 03.",
        "",
        "Update `experiments/03_loso_model_comparison/experiment.yaml`:",
        "",
        "```yaml",
        "density:",
        f"  kernel: \"{best['kernel_type']}\"",
    ])

    if kernel_type == "gaussian":
        lines.append(f"  sigma_sec: {best['sigma_sec']}")
    elif kernel_type == "skewed_gaussian":
        lines.append(f"  sigma_left_sec: {best['sigma_left_sec']}")
        lines.append(f"  sigma_right_sec: {best['sigma_right_sec']}")
    elif kernel_type == "cosine":
        lines.append(f"  half_width_sec: {best['sigma_sec']}")

    data_dir_map = {
        "gaussian": f"gaussian_sigma{int(best['sigma_sec']*1000)}ms",
        "skewed_gaussian": f"skewed_l{int(best['sigma_left_sec']*1000)}ms_r{int(best['sigma_right_sec']*1000)}ms",
        "cosine": f"cosine_half{int(best['sigma_sec']*1000)}ms",
    }
    data_dir = data_dir_map.get(kernel_type, "unknown")
    lines.append(f'  data_dir: "{data_dir}"')
    lines.append("```")

    with report_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_file


def main() -> None:
    cfg = load_config()
    exp_dir = Path(__file__).parent.parent
    runs_dir = exp_dir / "runs"

    if not runs_dir.exists():
        print("Error: runs/ directory does not exist")
        print("Please run scripts/02_train.py first")
        return

    # Collect results
    results = collect_kernel_results(runs_dir, cfg)

    if not results:
        print("Error: No results found in runs/")
        return

    # Find best kernel
    best_kernel = find_best_kernel(results)

    # Compile summary
    summary = {
        "experiment": cfg["name"],
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "kernels_tested": len(results),
        "best_kernel": best_kernel,
        "results": results,
    }

    # Save to result folder
    result_file = save_results_to_result_folder(summary, exp_dir)
    print(f"Results saved to: {result_file}")

    # Generate markdown report
    report_file = generate_markdown_report(summary, exp_dir)
    print(f"Report saved to: {report_file}")

    # Print summary
    print(f"\n{'='*60}")
    print("Density Kernel Comparison Summary")
    print(f"{'='*60}")
    print(f"Kernels tested: {len(results)}")
    print(f"\nBest Kernel: {best_kernel['kernel_name']} ({best_kernel['kernel_id']})")
    print(f"  Test Count MAE: {best_kernel['test_count_mae']:.4f}")
    print(f"  Test Count MAE (pos): {best_kernel['test_count_mae_pos']:.4f}")


if __name__ == "__main__":
    main()
