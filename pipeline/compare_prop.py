"""传播方法对比：标签传播(Zhou2004 LP) vs 最近原型基线(proto)，表征固定走缓存嵌入。

设计要点（不破坏已有交付）：
  - proto 行用 run_self_training 直接训练，【不走 run_route】，故不会覆盖
    result_*.json / classifier_*.pt（提交用的 LP 模型）。
  - LP 行直接读已有 comparison.csv，不重算。
  - 结果写到【新文件】comparison_propagation.csv。
  - 嵌入走缓存（use_cache=True），不重训 AE/SimCLR。

注意：proto 的置信度来自 softmax(余弦/0.1)，刻度比 LP 概率更尖；这里沿用同一
cfg.tau_prop 仅为「下游完全相同、只换传播算法」的公平对照，置信度数值不宜跨方法直接比。

用法：
    python -m pipeline.compare_prop
"""
from __future__ import annotations

import os

import pandas as pd

from core.config import CFG
from propagation.propagate import nearest_prototype, pseudo_label_accuracy, select_confident
from pipeline.run import get_embeddings, prepare_data
from train.train_classifier import run_self_training
from core.utils import ensure_dir, get_device


def main():
    cfg = CFG
    device = get_device()
    data = prepare_data(cfg)

    rows = []
    for route in ["pixel", "ae", "simclr"]:
        emb = get_embeddings(route, data["X_train"], cfg, use_cache=True, device=device)
        pseudo, conf = nearest_prototype(emb, data["labeled_idx"], data["labeled_y"], cfg)
        sel_idx, _ = select_confident(pseudo, conf, cfg.tau_prop, data["unlabeled_idx"])
        cov = len(sel_idx) / len(data["unlabeled_idx"])
        pacc = pseudo_label_accuracy(pseudo, data["y_train"], sel_idx)   # 仅诊断
        res = run_self_training(
            data["X_train"], data["y_train"],
            data["labeled_idx"], data["labeled_y"], data["unlabeled_idx"],
            pseudo, conf, data["val_loader"], data["test_loader"],
            cfg, device, verbose=False)
        rows.append(dict(
            propagation="proto", route=route,
            prop_pseudo_acc=round(pacc, 4), prop_coverage=round(cov, 4),
            best_round=res["best_round"],
            best_val=round(res["best_val"], 4),
            best_test=round(res["best_test"], 4),
        ))
        print(f"[proto] {route}: cov={cov:.3f} pacc={pacc:.4f} "
              f"-> best_round={res['best_round']} val={res['best_val']:.4f} "
              f"test={res['best_test']:.4f}")

    proto_df = pd.DataFrame(rows)

    # 合并已有 LP 结果（直接读 comparison.csv，不重算）
    lp_path = os.path.join(cfg.out_dir, "comparison.csv")
    if os.path.exists(lp_path):
        lp_df = pd.read_csv(lp_path)
        lp_df.insert(0, "propagation", "lp")
        combined = pd.concat([lp_df, proto_df], ignore_index=True)
    else:
        print(f"[警告] 未找到 {lp_path}，仅输出 proto 行")
        combined = proto_df

    ensure_dir(cfg.out_dir)
    out = os.path.join(cfg.out_dir, "comparison_propagation.csv")
    combined.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n=== 传播方法对比：LP(Zhou2004) vs proto(最近原型)，best_* 按 val 选 ===")
    print(combined.to_string(index=False))
    print(f"\n已保存: {out}")


if __name__ == "__main__":
    main()
