"""Stage 2：在嵌入空间里把 10 个锚点的标签传播为伪标签。

主方法：Zhou et al. (2004) “Learning with Local and Global Consistency” 标签传播
    F* 迭代： F <- alpha * S * F + (1 - alpha) * Y,   S = D^{-1/2} W D^{-1/2}
基线：最近原型（每类仅 1 个锚点，直接按余弦最近邻分配）。

kNN 图用 faiss 构建（5 万点秒级）。
"""
from __future__ import annotations

from typing import Tuple

import faiss
import numpy as np
import scipy.sparse as sp

from core.config import CFG, Config
from core.utils import numpy_softmax


def l2_normalize(emb: np.ndarray) -> np.ndarray:
    emb = np.ascontiguousarray(emb, dtype=np.float32)
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.clip(norm, 1e-8, None)


def build_knn_graph(emb_norm: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """返回每个点的 k 近邻 (neighbors[N,k], sims[N,k])，已剔除自身。余弦相似度（内积）。"""
    n, d = emb_norm.shape
    index = faiss.IndexFlatIP(d)
    index.add(emb_norm)
    sims, nbrs = index.search(emb_norm, k + 1)   # 含自身
    # 剔除自身列（IP 下自身相似度最高，通常在第 0 列）
    neighbors = np.empty((n, k), dtype=np.int64)
    weights = np.empty((n, k), dtype=np.float32)
    for i in range(n):
        row_n, row_s = nbrs[i], sims[i]
        keep = row_n != i
        if keep.sum() >= k:
            neighbors[i] = row_n[keep][:k]
            weights[i] = row_s[keep][:k]
        else:  # 极少数自身未在结果中，直接丢首列
            neighbors[i] = row_n[1:k + 1]
            weights[i] = row_s[1:k + 1]
    return neighbors, weights


def _affinity_matrix(neighbors: np.ndarray, weights: np.ndarray, n: int) -> sp.csr_matrix:
    k = neighbors.shape[1]
    rows = np.repeat(np.arange(n), k)
    cols = neighbors.ravel()
    data = np.clip(weights.ravel(), 0.0, None)        # 余弦可能为负，截断到 0
    W = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    W = W.maximum(W.T)                                # 对称化
    return W


def _normalized_affinity(W: sp.csr_matrix) -> sp.csr_matrix:
    deg = np.asarray(W.sum(axis=1)).ravel()
    d_inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    d_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    D = sp.diags(d_inv_sqrt)
    return D @ W @ D                                  # S = D^{-1/2} W D^{-1/2}


def label_propagation(emb: np.ndarray, labeled_idx: np.ndarray, labeled_y: np.ndarray,
                      cfg: Config = CFG) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zhou(2004) 标签传播。返回 (pseudo[N], confidence[N], F[N,C])。

    confidence 为按行归一化后的最大类概率；孤立点置信度趋近 0。
    """
    n = emb.shape[0]
    C = cfg.num_classes
    emb_norm = l2_normalize(emb)

    neighbors, weights = build_knn_graph(emb_norm, cfg.knn_k)
    W = _affinity_matrix(neighbors, weights, n)
    S = _normalized_affinity(W).tocsr()

    Y = np.zeros((n, C), dtype=np.float32)
    Y[labeled_idx, labeled_y] = 1.0

    F = Y.copy()
    alpha = cfg.lp_alpha
    for _ in range(cfg.lp_iters):
        F = alpha * (S @ F) + (1.0 - alpha) * Y

    row_sum = F.sum(axis=1, keepdims=True)
    probs = F / np.clip(row_sum, 1e-12, None)
    pseudo = probs.argmax(axis=1).astype(np.int64)
    confidence = probs.max(axis=1).astype(np.float32)
    confidence[row_sum.ravel() <= 1e-12] = 0.0        # 未被传播到的孤立点
    return pseudo, confidence, F


def nearest_prototype(emb: np.ndarray, labeled_idx: np.ndarray, labeled_y: np.ndarray,
                      cfg: Config = CFG) -> Tuple[np.ndarray, np.ndarray]:
    """基线：每类 1 个锚点作原型，按余弦相似度分配。返回 (pseudo[N], confidence[N])。"""
    emb_norm = l2_normalize(emb)
    protos = emb_norm[labeled_idx]                    # (C, d)，按 labeled_y 对应类别
    sims = emb_norm @ protos.T                        # (N, C) 列序 = labeled 顺序
    order = np.argsort(labeled_y)                     # 重排到类别 0..C-1
    sims = sims[:, order]
    probs = numpy_softmax(sims / 0.1, axis=1)
    pseudo = probs.argmax(axis=1).astype(np.int64)
    confidence = probs.max(axis=1).astype(np.float32)
    return pseudo, confidence


def select_confident(pseudo: np.ndarray, confidence: np.ndarray, tau: float,
                     candidate_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """在候选（无标签）索引中挑出置信度 >= tau 的，返回 (selected_idx, selected_pseudo)。"""
    conf_c = confidence[candidate_idx]
    keep = conf_c >= tau
    sel = candidate_idx[keep]
    return sel, pseudo[sel]


def pseudo_label_accuracy(pseudo: np.ndarray, y_true: np.ndarray,
                          idx: np.ndarray) -> float:
    """诊断用：选中子集上的伪标签准确率。

    【仅用于报告分析，绝不可回流到训练或超参选择。】
    """
    if len(idx) == 0:
        return float("nan")
    return float((pseudo[idx] == y_true[idx]).mean())


if __name__ == "__main__":
    # 随机嵌入下的接口冒烟（非真实精度）
    rng = np.random.default_rng(0)
    N = 2000
    emb = rng.standard_normal((N, 16)).astype(np.float32)
    lab_idx = np.arange(10)
    lab_y = np.arange(10)
    p, c, F = label_propagation(emb, lab_idx, lab_y, CFG.override(knn_k=10))
    print("pseudo", p.shape, "conf range", round(float(c.min()), 3), round(float(c.max()), 3))
    sel, sy = select_confident(p, c, 0.2, np.arange(10, N))
    print("selected", len(sel))
