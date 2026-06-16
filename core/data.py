"""数据层：逐字复现作业 bootstrap 的 MNIST 划分，并构造半监督所需的各种数据视图。

关键：ANCHOR_INDICES 是“划分之后”train_dataset 内的位置，所以下面的
seed / Resize(32) / random_split([50000,10000]) / generator(seed) 必须与作业完全一致。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.datasets.mnist import read_image_file, read_label_file

from core.config import ANCHOR_INDICES, CFG, Config
from core.utils import seed_torch


_TRAIN_IMAGE_FILE = "train-images-idx3-ubyte"
_TRAIN_LABEL_FILE = "train-labels-idx1-ubyte"
_TEST_IMAGE_FILE = "t10k-images-idx3-ubyte"
_TEST_LABEL_FILE = "t10k-labels-idx1-ubyte"


class LocalMNISTDataset(Dataset):
    """Dataset wrapper for raw IDX files placed directly under cfg.data_root."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor,
                 transform=None, target_transform=None):
        self.data = images
        self.targets = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return int(self.targets.numel())

    def __getitem__(self, index: int):
        image = Image.fromarray(self.data[index].numpy(), mode="L")
        target = int(self.targets[index])
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


def _has_flat_mnist(root: str) -> bool:
    base = Path(root)
    required = [
        _TRAIN_IMAGE_FILE, _TRAIN_LABEL_FILE,
        _TEST_IMAGE_FILE, _TEST_LABEL_FILE,
    ]
    return all((base / name).is_file() for name in required)


def _flat_mnist_paths(root: str, train: bool) -> Tuple[Path, Path]:
    base = Path(root)
    if train:
        return base / _TRAIN_IMAGE_FILE, base / _TRAIN_LABEL_FILE
    return base / _TEST_IMAGE_FILE, base / _TEST_LABEL_FILE


def _load_flat_mnist(root: str, train: bool, transform) -> LocalMNISTDataset:
    image_path, label_path = _flat_mnist_paths(root, train)
    images = read_image_file(str(image_path))
    labels = read_label_file(str(label_path))
    return LocalMNISTDataset(images, labels, transform=transform)


def _load_mnist(root: str, train: bool, transform) -> Dataset:
    if _has_flat_mnist(root):
        return _load_flat_mnist(root, train, transform)
    return MNIST(root=root, train=train, download=True, transform=transform)


# —— 作业给定的 transform（请勿改动）——
def _build_transform(cfg: Config) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(cfg.img_size),   # 32
        transforms.ToTensor(),
    ])


def load_datasets(cfg: Config = CFG) -> Tuple[Dataset, Subset, Subset, Dataset]:
    """复现作业划分。返回 (mnist_train_full, train_dataset, val_dataset, mnist_test)。

    划分只由 generator 的种子决定，与 data_root 无关。
    """
    seed_torch(cfg.seed)
    transform = _build_transform(cfg)

    mnist_train = _load_mnist(cfg.data_root, train=True, transform=transform)
    mnist_test = _load_mnist(cfg.data_root, train=False, transform=transform)

    # split train/val subset —— 与作业一致：generator 显式以 seed 播种
    generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset, val_dataset = random_split(
        mnist_train, list(cfg.split_sizes), generator=generator
    )
    return mnist_train, train_dataset, val_dataset, mnist_test


def materialize(dataset: Dataset, cfg: Config = CFG) -> Tuple[torch.Tensor, torch.Tensor]:
    """把一个数据集按顺序物化成张量 (X[N,1,H,W] float32, y[N] long)。

    顺序与 dataset 的索引一致，因此 X[i] 对应 dataset[i]。
    """
    loader = DataLoader(dataset, batch_size=cfg.emb_batch_size, shuffle=False,
                        num_workers=cfg.num_workers)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    X = torch.cat(xs, dim=0).contiguous()
    y = torch.cat(ys, dim=0).contiguous().long()
    return X, y


def split_labeled_unlabeled(
    y_true: torch.Tensor, cfg: Config = CFG, verify: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按 ANCHOR_INDICES 切出有标签(10)/无标签(其余)索引。

    返回 (labeled_idx[10], labeled_y[10], unlabeled_idx[N-10])，均为 numpy 数组。
    labeled_y 取自这 10 个锚点的真标签 —— 这是作业【允许】使用的标签。
    """
    n = y_true.numel()
    labeled_idx = np.asarray(ANCHOR_INDICES, dtype=np.int64)
    labeled_y = y_true.numpy()[labeled_idx]

    mask = np.ones(n, dtype=bool)
    mask[labeled_idx] = False
    unlabeled_idx = np.nonzero(mask)[0]

    if verify:
        verify_anchors(labeled_idx, labeled_y, cfg)
    return labeled_idx, labeled_y, unlabeled_idx


def verify_anchors(labeled_idx: np.ndarray, labeled_y: np.ndarray, cfg: Config = CFG) -> None:
    """关键 sanity check：确认我们复现的划分下，10 个锚点恰好覆盖 0-9 每类一张。

    若失败，说明本机复现的 random_split 顺序与作业不一致（须立即排查，否则索引无意义）。
    """
    labels = sorted(int(v) for v in labeled_y)
    print(f"[校验] 锚点索引 -> 真标签: "
          f"{dict(zip(labeled_idx.tolist(), labeled_y.tolist()))}")
    assert len(labeled_idx) == cfg.num_classes, \
        f"锚点数应为 {cfg.num_classes}，实际 {len(labeled_idx)}"
    assert labels == list(range(cfg.num_classes)), (
        f"锚点标签应为 0-9 每类一张，实际为 {labels}。"
        f"这说明本机 random_split 划分与作业不一致——请检查 torch 版本/种子。"
    )
    print("[校验] 通过：10 个锚点恰好每类一张，划分复现正确。")


# —— 给最终 CNN 分类器用的数据集：在物化张量上按索引取图 + 指定（伪）标签 ——
class IndexedTensorDataset(Dataset):
    """从物化张量 X 中按 indices 取图，配上给定 labels；可选轻量增广。"""

    def __init__(self, X: torch.Tensor, indices: np.ndarray, labels: np.ndarray,
                 transform: Optional[callable] = None):
        assert len(indices) == len(labels)
        self.X = X
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = torch.as_tensor(np.asarray(labels), dtype=torch.long)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        img = self.X[self.indices[i]]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i]


def make_classifier_loader(X: torch.Tensor, indices: np.ndarray, labels: np.ndarray,
                           cfg: Config = CFG, shuffle: bool = True,
                           transform: Optional[callable] = None) -> DataLoader:
    ds = IndexedTensorDataset(X, indices, labels, transform=transform)
    return DataLoader(ds, batch_size=cfg.cls_batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers, drop_last=False)


def tensor_eval_loader(X: torch.Tensor, y: torch.Tensor, cfg: Config = CFG) -> DataLoader:
    """给 val/test 用的评估 loader（不打乱）。"""
    ds = torch.utils.data.TensorDataset(X, y)
    return DataLoader(ds, batch_size=cfg.emb_batch_size, shuffle=False,
                      num_workers=cfg.num_workers)


if __name__ == "__main__":
    # 冒烟测试：复现划分并校验锚点
    _, train_dataset, val_dataset, mnist_test = load_datasets()
    print(f"train_dataset={len(train_dataset)}  val_dataset={len(val_dataset)}  "
          f"test={len(mnist_test)}")
    X_train, y_train = materialize(train_dataset)
    print(f"X_train shape={tuple(X_train.shape)}  dtype={X_train.dtype}  "
          f"range=[{X_train.min():.3f},{X_train.max():.3f}]")
    split_labeled_unlabeled(y_train, verify=True)
