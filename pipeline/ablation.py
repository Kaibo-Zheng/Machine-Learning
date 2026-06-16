"""消融表：表征(pixel/ae/simclr) × 阶段(仅标签传播 vs +自训练)。

直接读取 compare.py 产出的 result_{route}.json 与 comparison.csv，【不重复训练】：
  - 「仅传播」 = 自训练 round 0（在 10 真标签 + 传播伪标签上训一次 CNN）
  - 「+自训练」= 按 val 选出的最优轮
报告 val（选模型依据）与 test（仅汇报），以及传播覆盖率 / 伪标签准确率（诊断）。

用法：
    python -m pipeline.ablation
"""
from __future__ import annotations

import json
import os

import pandas as pd

from core.config import CFG
from core.utils import ensure_dir


def main():
    out_dir = CFG.out_dir
    comp_path = os.path.join(out_dir, "comparison.csv")
    comp = (pd.read_csv(comp_path).set_index("route")
            if os.path.exists(comp_path) else None)

    rows = []
    for route in ["pixel", "ae", "simclr"]:
        rp = os.path.join(out_dir, f"result_{route}.json")
        if not os.path.exists(rp):
            print(f"[跳过] 缺少 {rp}（请先跑 compare.py）")
            continue
        with open(rp, encoding="utf-8") as f:
            res = json.load(f)
        r0 = res["rounds"][0]                       # 仅传播（round 0）
        prop_acc = comp.loc[route, "prop_pseudo_acc"] if comp is not None else float("nan")
        prop_cov = comp.loc[route, "prop_coverage"] if comp is not None else float("nan")
        rows.append(dict(
            route=route,
            prop_pseudo_acc=prop_acc, prop_coverage=prop_cov,
            val_prop_only=round(r0["val_acc"], 4),
            test_prop_only=round(r0["test_acc"], 4),
            best_round=res["best_round"],
            val_self_train=round(res["best_val"], 4),
            test_self_train=round(res["best_test"], 4),
            test_gain=round(res["best_test"] - r0["test_acc"], 4),
        ))

    df = pd.DataFrame(rows)
    ensure_dir(out_dir)
    out_csv = os.path.join(out_dir, "ablation.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("=== 消融：表征 × (仅传播 / +自训练) ===")
    print(df.to_string(index=False))
    print(f"\n已保存: {out_csv}")


if __name__ == "__main__":
    main()
