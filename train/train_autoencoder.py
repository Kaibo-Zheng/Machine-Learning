"""路线 A 的表征：卷积自编码机（无标签，重建损失）。

在全部 5 万张训练图上训练，取 encoder 瓶颈作为嵌入。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from core.config import CFG, Config
from core.models import ConvAutoencoder
from core.utils import get_device


def train_autoencoder(X_all: torch.Tensor, cfg: Config = CFG,
                      device: torch.device = None, verbose: bool = True) -> ConvAutoencoder:
    """在全部训练图 X_all (N,1,32,32) 上训练自编码机（不使用任何标签）。"""
    device = device or get_device()
    model = ConvAutoencoder(cfg.embed_dim).to(device)
    loader = DataLoader(TensorDataset(X_all), batch_size=cfg.ae_batch_size,
                        shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.ae_lr)

    model.train()
    for epoch in range(cfg.ae_epochs):
        running = 0.0
        it = tqdm(loader, disable=not verbose, desc=f"AE epoch {epoch+1}/{cfg.ae_epochs}")
        for (x,) in it:
            x = x.to(device, non_blocking=True)
            recon = model(x)
            loss = F.mse_loss(recon, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            it.set_postfix(loss=f"{loss.item():.4f}")
        if verbose:
            print(f"[AE] epoch {epoch+1}: recon_mse={running/len(loader.dataset):.5f}")
    return model


@torch.no_grad()
def extract_embeddings(model, X_all: torch.Tensor, cfg: Config = CFG,
                       device: torch.device = None) -> np.ndarray:
    """对任意带 .encode(x) 的模型，按 X_all 顺序抽取嵌入，返回 (N, embed_dim) float32。

    顺序与 X_all 一致，故 emb[i] 对应 train_dataset[i]，可直接用 ANCHOR_INDICES 索引。
    """
    device = device or get_device()
    model.eval().to(device)
    out = []
    for i in range(0, X_all.size(0), cfg.emb_batch_size):
        xb = X_all[i:i + cfg.emb_batch_size].to(device, non_blocking=True)
        h = model.encode(xb)
        out.append(h.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


if __name__ == "__main__":
    # 快速冒烟：少量 epoch
    from core.data import load_datasets, materialize
    cfg = CFG.override(ae_epochs=1)
    _, train_dataset, _, _ = load_datasets(cfg)
    X, _ = materialize(train_dataset, cfg)
    m = train_autoencoder(X, cfg)
    emb = extract_embeddings(m, X, cfg)
    print("emb shape:", emb.shape)
