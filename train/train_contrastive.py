"""路线 B 的表征：SimCLR 自监督对比学习（无标签）。

要点：
  - 双视图增广刻意【不含翻转】——数字不是翻转不变的（2/5/6/9 等翻转会变成别的类）。
  - NT-Xent 损失，下游使用 backbone 的表征 h（不经投影头）。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from core.config import CFG, Config
from core.models import SimCLRNet
from core.utils import get_device


def build_simclr_augment(img_size: int = 32):
    """张量级双视图增广（输入已是 (1,32,32) 的 [0,1] 张量）。"""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), antialias=True),
        transforms.RandomApply([transforms.RandomRotation(20)], p=0.5),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.5),
        # 故意不加 Horizontal/Vertical Flip
    ])


class TwoViewsDataset(Dataset):
    """每次取一张图，返回两个独立增广视图。"""

    def __init__(self, X: torch.Tensor, augment):
        self.X = X
        self.augment = augment

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, i: int):
        x = self.X[i]
        return self.augment(x), self.augment(x)


def nt_xent_loss(z: torch.Tensor, temperature: float) -> torch.Tensor:
    """NT-Xent（z 已 L2 归一化，形状 (2N, d)，正样本对为 (i, i+N)）。"""
    n2 = z.size(0)
    sim = (z @ z.t()) / temperature                       # (2N, 2N)
    sim.fill_diagonal_(float("-inf"))                     # 去掉自相似
    n = n2 // 2
    targets = (torch.arange(n2, device=z.device) + n) % n2
    return F.cross_entropy(sim, targets)


def train_simclr(X_all: torch.Tensor, cfg: Config = CFG,
                 device: torch.device = None, verbose: bool = True) -> SimCLRNet:
    """在全部训练图上训练 SimCLR（不使用任何标签）。"""
    device = device or get_device()
    model = SimCLRNet(cfg.embed_dim, cfg.proj_dim).to(device)
    ds = TwoViewsDataset(X_all, build_simclr_augment(cfg.img_size))
    loader = DataLoader(ds, batch_size=cfg.simclr_batch_size, shuffle=True,
                        num_workers=cfg.num_workers, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.simclr_lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.simclr_epochs)

    model.train()
    for epoch in range(cfg.simclr_epochs):
        running = 0.0
        it = tqdm(loader, disable=not verbose,
                  desc=f"SimCLR epoch {epoch+1}/{cfg.simclr_epochs}")
        for v1, v2 in it:
            x = torch.cat([v1, v2], dim=0).to(device, non_blocking=True)  # (2B,1,32,32)
            z = model(x)                                                  # 已归一化
            loss = nt_xent_loss(z, cfg.simclr_temp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
            it.set_postfix(loss=f"{loss.item():.4f}")
        sched.step()
        if verbose:
            print(f"[SimCLR] epoch {epoch+1}: ntxent={running/len(loader):.4f}")
    return model


if __name__ == "__main__":
    from core.data import load_datasets, materialize
    from train.train_autoencoder import extract_embeddings
    cfg = CFG.override(simclr_epochs=1, simclr_batch_size=256)
    _, train_dataset, _, _ = load_datasets(cfg)
    X, _ = materialize(train_dataset, cfg)
    m = train_simclr(X, cfg)
    emb = extract_embeddings(m, X, cfg)   # 复用同一抽取函数（模型都有 .encode）
    print("emb shape:", emb.shape)
