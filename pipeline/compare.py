"""对比三条表征路线（像素基线 / 自编码机 / SimCLR），汇总成对比表。

两条路线共用同一份物化数据与同一套超参，保证对比公平。
用法：
    python -m pipeline.compare            # 完整对比（默认超参）
    python -m pipeline.compare --quick    # 小 epoch 冒烟
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from core.config import CFG
from pipeline.run import build_cfg_from_args, get_embeddings, prepare_data, run_route
from propagation.propagate import (label_propagation, pseudo_label_accuracy, select_confident)
from core.utils import ensure_dir, get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--routes", nargs="+", default=["pixel", "ae", "simclr"])
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
    device = get_device()
    data = prepare_data(cfg)

    rows = []
    for route in args.routes:
        print(f"\n========== 路线: {route} ==========")
        # 传播质量诊断（第 0 轮、传播原始结果）：覆盖率 + 伪标签准确率
        emb = get_embeddings(route, data["X_train"], cfg,
                             use_cache=not args.no_cache, device=device)
        pseudo, conf, _ = label_propagation(
            emb, data["labeled_idx"], data["labeled_y"], cfg)
        sel_idx, _ = select_confident(pseudo, conf, cfg.tau_prop, data["unlabeled_idx"])
        diag = pseudo_label_accuracy(pseudo, data["y_train"], sel_idx)
        coverage = len(sel_idx) / len(data["unlabeled_idx"])

        summary = run_route(route, data, cfg, use_cache=not args.no_cache,
                            propagation=args.propagation, device=device)
        rows.append(dict(
            route=route,
            prop_pseudo_acc=round(diag, 4),       # 诊断：传播伪标签准确率
            prop_coverage=round(coverage, 4),     # 诊断：tau 下覆盖率
            best_round=summary["best_round"],
            best_val=round(summary["best_val"], 4),
            best_test=round(summary["best_test"], 4),
        ))

    df = pd.DataFrame(rows)
    ensure_dir(cfg.out_dir)
    out_csv = os.path.join(cfg.out_dir, "comparison.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("\n=== 路线对比（best_* 为按 val 选出的最终结果，test 仅汇报）===")
    print(df.to_string(index=False))
    print(f"\n已保存: {out_csv}")


if __name__ == "__main__":
    main()
