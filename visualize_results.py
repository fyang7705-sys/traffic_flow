import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_overall_metrics(json_path: Path) -> Dict[str, float]:
    with json_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict) or "overall" not in obj or not isinstance(obj["overall"], dict):
        raise ValueError(f"{json_path} 格式不符合预期：需要 {{'overall': {{...}}}}")

    overall = obj["overall"]
    metrics: Dict[str, float] = {}
    for k, v in overall.items():
        if isinstance(v, (int, float)):
            metrics[k] = float(v)
    return metrics


def collect_results(result_dir: Path) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for p in sorted(result_dir.glob("*.json")):
        model_name = p.stem
        try:
            results[model_name] = load_overall_metrics(p)
        except Exception as e:
            print(f"[跳过] {p.name}: {e}")
    if not results:
        raise RuntimeError(f"在目录 {result_dir} 下未找到可用的 result json")
    return results


def to_table(results: Dict[str, Dict[str, float]], metric_order: List[str] | None = None):
    import pandas as pd

    df = pd.DataFrame(results).T  # rows=model, cols=metric
    if metric_order is not None:
        cols = [c for c in metric_order if c in df.columns] + [c for c in df.columns if c not in metric_order]
        df = df[cols]
    return df


def plot_radar(df, out_dir: Path, metrics: List[str], normalize: str = "minmax"):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.projections.polar import PolarAxes
    from typing import cast

    use = [m for m in metrics if m in df.columns]
    if len(use) < 3:
        print("[跳过] 雷达图至少需要 3 个指标")
        return

    values = df[use].copy()

    # 所有指标按“越小越好”处理，做归一化后“越大越好”以便雷达图可读
    if normalize == "minmax":
        vmin = values.min(axis=0)
        vmax = values.max(axis=0)
        denom = (vmax - vmin).replace(0, 1.0)
        norm = (vmax - values) / denom  # best->1
    else:
        mean = values.mean(axis=0)
        std = values.std(axis=0).replace(0, 1.0)
        norm = (mean - values) / std

    labels = use
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(7, 7))
    ax = cast(PolarAxes, plt.subplot(111, polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)

    for model in norm.index:
        vals = norm.loc[model].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1.6, label=model)
        ax.fill(angles, vals, alpha=0.08)

    ax.set_title("Overall metrics")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))
    plt.tight_layout()

    out_path = out_dir / "compare_radar_12_12.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[保存] {out_path}")


def main():
    parser = argparse.ArgumentParser(description="可视化对比 result/*.json 的 overall 指标（仅雷达图）")
    parser.add_argument("--result_dir", type=str, default="result", help="result 目录（包含多个 *.json）")
    parser.add_argument("--out_dir", type=str, default="result_vis", help="输出图片/表格目录")
    parser.add_argument("--metrics", type=str, default="mae,mse,rmse,mape,wmape", help="要绘制的指标，逗号分隔")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = collect_results(result_dir)

    metric_list = [m.strip() for m in args.metrics.split(",") if m.strip()]

    try:
        import pandas as pd  # noqa: F401
        import matplotlib  # noqa: F401
    except Exception:
        print("缺少依赖：需要 pandas + matplotlib。")
        print("可用 pip 安装：pip install pandas matplotlib")
        return

    df = to_table(results, metric_order=metric_list)

    # 仍然保存一份表格，便于查看/复现实验
    csv_path = out_dir / "overall_metrics_12_12.csv"
    df.to_csv(csv_path, float_format="%.6f")
    print(f"[保存] {csv_path}")

    plot_radar(df, out_dir, metric_list)


if __name__ == "__main__":
    main()
