"""中央配置：所有超参数、路径、固定常量集中在此，便于复现与在报告里说明。

合规要点（务必遵守作业约束）：
  - SEED 与划分逻辑必须与作业给定的 bootstrap 完全一致，否则 ANCHOR_INDICES 失效。
  - train_dataset 中除了 ANCHOR_INDICES 这 10 张，其余图片的真标签【一律不得用于训练或选超参】。
  - val_dataset 的标签【允许】用于选择超参数（作业只限制了 train_dataset）。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List


# 作业给定：train_dataset 中允许使用标签的 10 张图片索引（每类各 1 张）。
# 这些索引是“划分之后”train_dataset 内的位置，因此划分必须逐字复现。
ANCHOR_INDICES: List[int] = [
    1173, 3336, 12529, 12785, 12979, 17351, 27048, 40579, 43128, 46498
]


@dataclass
class Config:
    # —— 固定常量（来自作业，请勿改动）——
    seed: int = 42
    img_size: int = 32          # 作业把 MNIST Resize 到了 32（非原生 28）
    num_classes: int = 10
    split_sizes: tuple = (50000, 10000)   # random_split(mnist_train, [50000, 10000])

    # —— 路径 ——
    data_root: str = "./data"   # MNIST 下载目录（不影响划分，划分只由 generator 种子决定）
    out_dir: str = "./result"   # 缓存嵌入 / 结果表 / 可视化
    model: str = "./model"      # 模型权重（encoder_*.pt / classifier_*.pt），重新训练会覆盖同名文件

    # —— 表征维度 ——
    embed_dim: int = 128        # backbone 输出维度（AE 瓶颈 = SimCLR 表征 h）
    proj_dim: int = 64          # SimCLR 投影头输出维度 z

    # —— DataLoader ——
    # Windows 下多进程 DataLoader 易出问题；默认 0 最稳，可按需调大（入口均有 __main__ 保护）。
    num_workers: int = 0
    emb_batch_size: int = 512   # 抽嵌入（按序、不打乱）

    # —— 自编码机 ——
    ae_epochs: int = 30
    ae_batch_size: int = 256
    ae_lr: float = 1e-3

    # —— SimCLR 对比学习 ——
    simclr_epochs: int = 150
    simclr_batch_size: int = 512
    simclr_lr: float = 1e-3
    simclr_temp: float = 0.5

    # —— 标签传播（Zhou et al. 2004 局部全局一致性）——
    knn_k: int = 15
    lp_alpha: float = 0.99
    lp_iters: int = 50

    # —— CNN 分类器 + 自训练 ——
    cls_epochs: int = 30
    cls_batch_size: int = 256
    cls_lr: float = 1e-3
    cls_weight_decay: float = 5e-4
    # 两个阶段的置信度阈值刻度不同，必须分开：
    #   - 标签传播的概率较"软"（经验上多在 0.5–0.9），阈值取 ~0.7。
    #   - CNN softmax 往往过度自信（常 >0.95），自训练新增样本取高阈 ~0.95。
    tau_prop: float = 0.7       # 标签传播阶段的伪标签置信度阈值（仅在 val 上调）
    tau_self: float = 0.95      # 自训练阶段 CNN 预测的置信度阈值（仅在 val 上调）
    self_train_rounds: int = 2  # 自训练轮数（0 表示只用传播得到的伪标签训一次）

    # —— 诊断 ——
    # 仅用于报告分析（如统计伪标签准确率）。绝不可回流到训练或超参选择中。
    diagnostic_use_true_labels: bool = True

    def override(self, **kwargs) -> "Config":
        """返回一个覆盖了部分字段的新配置（用于 argparse 覆盖默认值）。"""
        valid = {k: v for k, v in kwargs.items() if v is not None and hasattr(self, k)}
        return replace(self, **valid)


# 全局默认配置；脚本可基于它 .override(...)
CFG = Config()
