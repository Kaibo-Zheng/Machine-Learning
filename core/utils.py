"""公共工具：随机种子、设备、评估、检查点等。"""
from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager
from typing import Tuple

import numpy as np
import torch


def seed_torch(seed: int = 42) -> None:
    """逐字复用作业给定的种子设置，保证实验可复现。"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # 禁止 hash 随机化
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device) -> float:
    """在带标签的 loader 上计算分类准确率。loader 产出 (x, y)。"""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def save_ckpt(state: dict, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    torch.save(state, path)


def load_ckpt(path: str, map_location=None) -> dict:
    return torch.load(path, map_location=map_location)


@contextmanager
def timer(name: str = ""):
    t0 = time.time()
    yield
    dt = time.time() - t0
    print(f"[计时] {name}: {dt:.1f}s")


def numpy_softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)
