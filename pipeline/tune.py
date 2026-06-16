"""在 val 上选超参（合规：test 仅汇报，绝不参与选择）。

对某条路线的【缓存嵌入】扫描标签传播 / 自训练超参：
  - knn_k             : kNN 图邻居数（影响传播图，需对每个 k 重算一次传播）
  - tau_prop          : 传播伪标签置信度阈值（同一传播结果上只改阈值，几乎零成本）
  - self_train_rounds : 自训练轮数。单次以【最大轮】运行，run_self_training 内部
                        会按 val 选最优轮，故无需对该维度单独网格。

用法：
    python -m pipeline.tune --route simclr                         # 默认网格
    python -m pipeline.tune --route ae --knn_k 10 15 20 --tau_prop 0.6 0.7 0.8 --rounds 2
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from core.config import CFG
from propagation.propagate import label_propagation, pseudo_label_accuracy, select_confident
from pipeline.run import get_embeddings, prepare_data
from train.train_classifier import run_self_training
from core.utils import ensure_dir, get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--route", choices=["ae", "simclr", "pixel"], default="simclr")
    p.add_argument("--knn_k", type=int, nargs="+", default=[10, 15, 20])
    p.add_argument("--tau_prop", type=float, nargs="+", default=[0.6, 0.7, 0.8])
    p.add_argument("--rounds", type=int, default=2,
                   help="每个配置以该最大自训练轮数运行；内部按 val 选最优轮")
    p.add_argument("--cls_epochs", type=int, default=None)
    args = p.parse_args()

    device = get_device()
    base = CFG.override(self_train_rounds=args.rounds, cls_epochs=args.cls_epochs)
    data = prepare_data(base)
    emb = get_embeddings(args.route, data["X_train"], base, use_cache=True, device=device)

    rows = []
    for k in args.knn_k:
        cfg_k = base.override(knn_k=k)
        # 传播只依赖 knn_k：每个 k 只算一次，供所有 tau_prop 复用。
        pseudo, conf, _ = label_propagation(
            emb, data["labeled_idx"], data["labeled_y"], cfg_k)
        for tau in args.tau_prop:
            cfg_run = cfg_k.override(tau_prop=tau)
            sel_idx, _ = select_confident(pseudo, conf, tau, data["unlabeled_idx"])
            cov = len(sel_idx) / len(data["unlabeled_idx"])
            pacc = pseudo_label_accuracy(pseudo, data["y_train"], sel_idx)  # 仅诊断
            res = run_self_training(
                data["X_train"], data["y_train"],
                data["labeled_idx"], data["labeled_y"], data["unlabeled_idx"],
                pseudo, conf, data["val_loader"], data["test_loader"],
                cfg_run, device, verbose=False)
            rows.append(dict(
                route=args.route, knn_k=k, tau_prop=tau,
                prop_coverage=round(cov, 4), prop_pseudo_acc=round(pacc, 4),
                best_round=res["best_round"],
                best_val=round(res["best_val"], 4),
                best_test=round(res["best_test"], 4),
            ))
            print(f"[扫描] knn_k={k} tau_prop={tau} cov={cov:.3f} "
                  f"pacc={pacc:.4f} -> best_round={res['best_round']} "
                  f"val={res['best_val']:.4f} test={res['best_test']:.4f}")

    df = pd.DataFrame(rows).sort_values("best_val", ascending=False).reset_index(drop=True)
    ensure_dir(base.out_dir)
    out_csv = os.path.join(base.out_dir, f"tune_{args.route}.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    best = df.iloc[0]   # 选择只看 val
    print("\n=== 扫描结果（按 best_val 降序；选择只看 val，test 仅汇报）===")
    print(df.to_string(index=False))
    print(f"\n>>> 按 val 选中: knn_k={int(best.knn_k)} tau_prop={best.tau_prop} "
          f"self_train_round={int(best.best_round)}  "
          f"val={best.best_val:.4f}  (汇报 test={best.best_test:.4f})")
    print(f"已保存: {out_csv}")


if __name__ == "__main__":
    main()
