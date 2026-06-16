"""嵌入空间可视化：对三条路线的嵌入做 t-SNE，按真标签着色，标出 10 个锚点。

【合规】真标签仅用于可视化着色这一【诊断】用途，绝不回流到训练或选超参
（与 propagate.pseudo_label_accuracy 同级，受 config.diagnostic_use_true_labels 精神约束）。

用法：
    python -m visualization.tsne_viz                # 默认抽样 6000 点 + 10 锚点
    python -m visualization.tsne_viz --n 8000 --perplexity 40
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from core.config import ANCHOR_INDICES, CFG
from pipeline.run import get_embeddings, prepare_data
from core.utils import ensure_dir, get_device


def run_tsne(emb: np.ndarray, seed: int, perplexity: float, pca_dim: int = 50) -> np.ndarray:
    """先 PCA 降到 pca_dim（统一像素/AE/SimCLR 的预处理），再 t-SNE 到 2D。"""
    if emb.shape[1] > pca_dim:
        emb = PCA(n_components=pca_dim, random_state=seed).fit_transform(emb)
    return TSNE(n_components=2, perplexity=perplexity, init="pca",
                random_state=seed).fit_transform(emb)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--routes", nargs="+", default=["pixel", "ae", "simclr"])
    p.add_argument("--n", type=int, default=6000, help="抽样点数（t-SNE 不宜全量 5 万）")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = get_device()
    data = prepare_data(CFG)
    y = data["y_train"]                 # 真标签，仅供着色（诊断）
    n_total = data["X_train"].size(0)

    rng = np.random.default_rng(args.seed)
    anchors = np.asarray(ANCHOR_INDICES)
    pool = np.setdiff1d(np.arange(n_total), anchors)
    sub = rng.choice(pool, size=min(args.n, len(pool)), replace=False)
    idx = np.concatenate([anchors, sub])     # 前 10 个是锚点
    yc = y[idx]

    fig, axes = plt.subplots(1, len(args.routes),
                             figsize=(6 * len(args.routes), 5.6), squeeze=False)
    axes = axes[0]
    sc = None
    for ax, route in zip(axes, args.routes):
        emb = get_embeddings(route, data["X_train"], CFG, use_cache=True, device=device)
        emb2d = run_tsne(emb[idx], args.seed, args.perplexity)
        sc = ax.scatter(emb2d[:, 0], emb2d[:, 1], c=yc, cmap="tab10", s=5, alpha=0.6)
        ax.scatter(emb2d[:10, 0], emb2d[:10, 1], c=yc[:10], cmap="tab10",
                   s=260, marker="*", edgecolors="black", linewidths=1.2, zorder=5)
        ax.set_title(f"{route}  (t-SNE)")
        ax.set_xticks([]); ax.set_yticks([])
        print(f"[t-SNE] {route} 完成")

    cbar = fig.colorbar(sc, ax=list(axes), ticks=range(10), fraction=0.02, pad=0.01)
    cbar.set_label("true digit (diagnostic only)")
    ensure_dir(CFG.out_dir)
    out = os.path.join(CFG.out_dir, "tsne_routes.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"已保存 t-SNE 图: {out}")
    ensure_dir("doc")
    doc_out = os.path.join("doc", "6.png")
    fig.savefig(doc_out, dpi=150, bbox_inches="tight")
    print(f"已保存 t-SNE 图: {doc_out}")


if __name__ == "__main__":
    main()
