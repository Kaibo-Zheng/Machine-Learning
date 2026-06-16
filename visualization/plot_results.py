"""把所有结果汇总成一张多面板图，供报告使用（读 result/ 的 CSV/JSON，不重训）。

面板：
  (a) 三路线最终 test：仅传播(round0) vs +自训练(best)        —— ablation.csv
  (b) 传播方法 test：LP(Zhou2004) vs proto(最近原型)          —— comparison_propagation.csv
  (c) 自训练逐轮 test_acc                                     —— result_{route}.json
  (d) SimCLR 超参扫描：val 随 knn_k × tau_prop（按此选超参）  —— tune_simclr.csv

用法：
    python -m visualization.plot_results
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 中文字体（Windows 11 自带 Microsoft YaHei）
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = "result"
DOC_OUT = "doc"
ROUTES = ["pixel", "ae", "simclr"]
NAVY = "#002554"


def annotate(ax, bars, fmt="{:.3f}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=8)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(len(ROUTES))
    w = 0.38

    # (a) 仅传播 vs +自训练（LP），test
    abl = pd.read_csv(os.path.join(OUT, "ablation.csv")).set_index("route")
    ax = axes[0, 0]
    b1 = ax.bar(x - w / 2, [abl.loc[r, "test_prop_only"] for r in ROUTES], w,
                label="仅传播 (round0)", color="#8da0cb")
    b2 = ax.bar(x + w / 2, [abl.loc[r, "test_self_train"] for r in ROUTES], w,
                label="+自训练 (best)", color=NAVY)
    annotate(ax, b1); annotate(ax, b2)
    ax.set_xticks(x); ax.set_xticklabels(ROUTES)
    ax.set_ylabel("test accuracy"); ax.set_ylim(0.8, 1.0)
    ax.set_title("(a) 三路线：仅传播 vs +自训练 (LP)")
    ax.legend(loc="lower right")

    # (b) LP vs proto, test
    cp = pd.read_csv(os.path.join(OUT, "comparison_propagation.csv"))
    piv = cp.pivot(index="route", columns="propagation", values="best_test").loc[ROUTES]
    ax = axes[0, 1]
    bl = ax.bar(x - w / 2, piv["lp"], w, label="LP (Zhou2004)", color=NAVY)
    bp = ax.bar(x + w / 2, piv["proto"], w, label="proto (最近原型)", color="#fc8d62")
    annotate(ax, bl); annotate(ax, bp)
    ax.set_xticks(x); ax.set_xticklabels(ROUTES)
    ax.set_ylabel("test accuracy"); ax.set_ylim(0.8, 1.0)
    ax.set_title("(b) 传播方法：LP vs 最近原型基线")
    ax.legend(loc="lower right")

    # (c) 自训练逐轮曲线
    ax = axes[1, 0]
    for r, color in zip(ROUTES, ["#8da0cb", "#66c2a5", NAVY]):
        with open(os.path.join(OUT, f"result_{r}.json"), encoding="utf-8") as f:
            rounds = json.load(f)["rounds"]
        xs = [d["round"] for d in rounds]
        ys = [d["test_acc"] for d in rounds]
        ax.plot(xs, ys, marker="o", label=r, color=color)
        for xx, yy in zip(xs, ys):
            ax.annotate(f"{yy:.3f}", (xx, yy), fontsize=8, va="bottom")
    ax.set_xlabel("自训练轮数 round"); ax.set_ylabel("test accuracy")
    ax.set_xticks([0, 1, 2])
    ax.set_title("(c) 自训练逐轮 test (LP)")
    ax.legend(loc="lower right")

    # (d) 超参扫描热力图（simclr val）
    ax = axes[1, 1]
    tune = pd.read_csv(os.path.join(OUT, "tune_simclr.csv"))
    hm = tune.pivot(index="knn_k", columns="tau_prop", values="best_val")
    im = ax.imshow(hm.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(hm.columns))); ax.set_xticklabels(hm.columns)
    ax.set_yticks(range(len(hm.index))); ax.set_yticklabels(hm.index)
    ax.set_xlabel("tau_prop"); ax.set_ylabel("knn_k")
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            ax.text(j, i, f"{hm.values[i, j]:.4f}", ha="center", va="center",
                    color="white", fontsize=9)
    ax.set_title("(d) SimCLR 超参扫描 (val，按此选超参)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="val accuracy")

    fig.tight_layout()
    out = os.path.join(OUT, "results_summary.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"已保存: {out}")
    os.makedirs(DOC_OUT, exist_ok=True)
    doc_out = os.path.join(DOC_OUT, "5.png")
    fig.savefig(doc_out, dpi=150, bbox_inches="tight")
    print(f"已保存: {doc_out}")


if __name__ == "__main__":
    main()
