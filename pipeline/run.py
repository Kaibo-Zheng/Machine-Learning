"""端到端跑单条表征路线：数据 -> 嵌入(可缓存) -> 标签传播 -> CNN + 自训练 -> val/test。

用法示例：
    python -m pipeline.run --route ae
    python -m pipeline.run --route simclr
    python -m pipeline.run --route ae --quick          # 小 epoch 冒烟
    python -m pipeline.run --route simclr --no-cache   # 改了嵌入超参后强制重训
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import torch

from core.config import CFG, Config
from core.data import (load_datasets, materialize, split_labeled_unlabeled,
                  tensor_eval_loader)
from propagation.propagate import label_propagation, nearest_prototype
from train.train_autoencoder import extract_embeddings, train_autoencoder
from train.train_classifier import run_self_training
from train.train_contrastive import train_simclr
from core.utils import ensure_dir, get_device, save_ckpt, timer


def prepare_data(cfg: Config = CFG) -> Dict:
    """物化数据并切出有/无标签索引、val/test 评估 loader。两条路线可共用。"""
    _, train_dataset, val_dataset, mnist_test = load_datasets(cfg)
    X_train, y_train = materialize(train_dataset, cfg)
    X_val, y_val = materialize(val_dataset, cfg)
    X_test, y_test = materialize(mnist_test, cfg)
    labeled_idx, labeled_y, unlabeled_idx = split_labeled_unlabeled(y_train, cfg)
    return dict(
        X_train=X_train, y_train=y_train.numpy(),
        val_loader=tensor_eval_loader(X_val, y_val, cfg),
        test_loader=tensor_eval_loader(X_test, y_test, cfg),
        labeled_idx=labeled_idx, labeled_y=labeled_y, unlabeled_idx=unlabeled_idx,
    )


def get_embeddings(route: str, X_train: torch.Tensor, cfg: Config = CFG,
                   use_cache: bool = True, device=None) -> np.ndarray:
    """得到某条路线的嵌入。route ∈ {ae, simclr, pixel}。ae/simclr 会缓存。"""
    device = device or get_device()
    if route == "pixel":   # 基线：原始像素拉平，无需训练
        return X_train.reshape(X_train.size(0), -1).numpy().astype(np.float32)

    cache = os.path.join(cfg.out_dir, f"emb_{route}.npy")
    if use_cache and os.path.exists(cache):
        print(f"[缓存] 载入嵌入 {cache}")
        return np.load(cache)

    with timer(f"训练表征({route})"):
        if route == "ae":
            model = train_autoencoder(X_train, cfg, device)
        elif route == "simclr":
            model = train_simclr(X_train, cfg, device)
        else:
            raise ValueError(f"未知 route: {route}")
    emb = extract_embeddings(model, X_train, cfg, device)
    ensure_dir(cfg.out_dir)
    ensure_dir(cfg.model)
    np.save(cache, emb)
    save_ckpt({"model": model.state_dict(), "route": route},
              os.path.join(cfg.model, f"encoder_{route}.pt"))
    print(f"[缓存] 已保存嵌入 {cache}  shape={emb.shape}")
    return emb


def run_route(route: str, data: Dict, cfg: Config = CFG, use_cache: bool = True,
              propagation: str = "lp", device=None) -> Dict:
    device = device or get_device()
    emb = get_embeddings(route, data["X_train"], cfg, use_cache, device)

    with timer(f"标签传播({propagation})"):
        if propagation == "lp":
            pseudo, conf, _ = label_propagation(
                emb, data["labeled_idx"], data["labeled_y"], cfg)
        elif propagation == "proto":
            pseudo, conf = nearest_prototype(
                emb, data["labeled_idx"], data["labeled_y"], cfg)
        else:
            raise ValueError(f"未知 propagation: {propagation}")

    res = run_self_training(
        data["X_train"], data["y_train"],
        data["labeled_idx"], data["labeled_y"], data["unlabeled_idx"],
        pseudo, conf, data["val_loader"], data["test_loader"], cfg, device)

    # 保存结果（json 去掉张量 state）与最优分类器权重
    ensure_dir(cfg.out_dir)
    ensure_dir(cfg.model)
    save_ckpt({"state": res["best_state"], "route": route},
              os.path.join(cfg.model, f"classifier_{route}.pt"))
    summary = dict(
        route=route, propagation=propagation,
        best_round=res["best_round"], best_val=res["best_val"],
        best_test=res["best_test"],
        rounds=[{k: v for k, v in r.items() if k != "state"} for r in res["rounds"]],
    )
    with open(os.path.join(cfg.out_dir, f"result_{route}.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def build_cfg_from_args(args) -> Config:
    cfg = CFG.override(
        tau_prop=args.tau_prop, tau_self=args.tau_self,
        knn_k=args.knn_k, self_train_rounds=args.self_train_rounds,
        ae_epochs=args.ae_epochs, simclr_epochs=args.simclr_epochs,
        cls_epochs=args.cls_epochs,
    )
    if args.quick:   # 冒烟：极小 epoch，便于快速打通流程
        cfg = cfg.override(ae_epochs=1, simclr_epochs=1, cls_epochs=2,
                           self_train_rounds=1)
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--route", choices=["ae", "simclr", "pixel"], default="ae")
    p.add_argument("--propagation", choices=["lp", "proto"], default="lp")
    p.add_argument("--tau_prop", type=float, default=None)
    p.add_argument("--tau_self", type=float, default=None)
    p.add_argument("--knn_k", type=int, default=None)
    p.add_argument("--self_train_rounds", type=int, default=None)
    p.add_argument("--ae_epochs", type=int, default=None)
    p.add_argument("--simclr_epochs", type=int, default=None)
    p.add_argument("--cls_epochs", type=int, default=None)
    p.add_argument("--no-cache", dest="no_cache", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    cfg = build_cfg_from_args(args)
    data = prepare_data(cfg)
    summary = run_route(args.route, data, cfg, use_cache=not args.no_cache,
                        propagation=args.propagation)
    print("\n=== 结果汇总 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
