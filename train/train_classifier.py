"""Stage 3–4：在（真 + 伪）标签上训练最终 CNN，并做自训练迭代。

超参（如阈值 tau、轮数、lr）只允许用 val_dataset 选择。test 仅用于最终汇报。
自训练：用上一轮 CNN 对无标签数据重新打分 -> 取高置信 -> 扩充训练集 -> 重训。
那 10 个锚点始终以【真标签】参与训练。
"""
from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from core.config import CFG, Config
from core.data import make_classifier_loader
from core.models import CNNClassifier
from core.utils import evaluate, get_device


def build_cls_augment(img_size: int = 32):
    """分类器训练的轻量增广（数字友好，无翻转）。"""
    return transforms.Compose([
        transforms.RandomCrop(img_size, padding=4, padding_mode="edge"),
        transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
    ])


def train_one_cnn(X_all: torch.Tensor, train_idx: np.ndarray, train_labels: np.ndarray,
                  val_loader: DataLoader, cfg: Config = CFG, device: torch.device = None,
                  augment: bool = True, verbose: bool = False) -> Tuple[dict, float]:
    """训练一个 CNN，按 val 准确率选最优 epoch。返回 (best_state_dict, best_val_acc)。"""
    device = device or get_device()
    tf = build_cls_augment(cfg.img_size) if augment else None
    loader = make_classifier_loader(X_all, train_idx, train_labels, cfg,
                                    shuffle=True, transform=tf)
    model = CNNClassifier(cfg.num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.cls_lr,
                           weight_decay=cfg.cls_weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.cls_epochs)

    best_val, best_state = -1.0, None
    for epoch in range(cfg.cls_epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        val_acc = evaluate(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"    epoch {epoch+1}/{cfg.cls_epochs}  val_acc={val_acc:.4f}")
    return best_state, best_val


@torch.no_grad()
def predict_probs(model: nn.Module, X: torch.Tensor, cfg: Config = CFG,
                  device: torch.device = None) -> np.ndarray:
    """对张量 X (M,1,32,32) 输出 softmax 概率 (M,C)。"""
    device = device or get_device()
    model.eval().to(device)
    out = []
    for i in range(0, X.size(0), cfg.emb_batch_size):
        xb = X[i:i + cfg.emb_batch_size].to(device, non_blocking=True)
        out.append(F.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def run_self_training(
    X_all: torch.Tensor, y_true: np.ndarray,
    labeled_idx: np.ndarray, labeled_y: np.ndarray, unlabeled_idx: np.ndarray,
    init_pseudo: np.ndarray, init_conf: np.ndarray,
    val_loader: DataLoader, test_loader: DataLoader,
    cfg: Config = CFG, device: torch.device = None, verbose: bool = True,
) -> Dict:
    """从传播得到的伪标签出发，做 cfg.self_train_rounds 轮自训练。

    返回包含每轮 (n_pseudo, pseudo_acc[诊断], val_acc, test_acc) 及按 val 选出的最优轮。
    """
    device = device or get_device()
    rounds: List[Dict] = []

    # 标签传播得到的伪标签/置信度作为固定底座（贯穿所有轮，不被丢弃）。
    base_pseudo = init_pseudo.astype(np.int64)
    base_conf = init_conf.astype(np.float32)
    # CNN 重新打分的结果（首轮尚无，全部置 0 表示“未给出”）。
    cnn_pseudo = np.zeros_like(base_pseudo)
    cnn_conf = np.zeros_like(base_conf)

    for r in range(cfg.self_train_rounds + 1):
        # 并集累积：底座(传播, tau_prop) ∪ CNN 高置信(tau_self)；重叠处优先用 CNN 标签。
        base_ok = base_conf[unlabeled_idx] >= cfg.tau_prop
        cnn_ok = cnn_conf[unlabeled_idx] >= cfg.tau_self
        keep = base_ok | cnn_ok
        sel_idx = unlabeled_idx[keep]
        use_cnn = cnn_conf[sel_idx] >= cfg.tau_self
        sel_lab = np.where(use_cnn, cnn_pseudo[sel_idx], base_pseudo[sel_idx])

        train_idx = np.concatenate([labeled_idx, sel_idx])
        train_lab = np.concatenate([labeled_y, sel_lab])

        diag_acc = (float((sel_lab == y_true[sel_idx]).mean()) if
                    (cfg.diagnostic_use_true_labels and len(sel_idx)) else float("nan"))
        if verbose:
            print(f"[自训练 round {r}] 选入伪标签 {len(sel_idx)} 张"
                  f"（覆盖率 {len(sel_idx)/len(unlabeled_idx):.1%}）"
                  f"  伪标签准确率(诊断)={diag_acc:.4f}")

        best_state, val_acc = train_one_cnn(
            X_all, train_idx, train_lab, val_loader, cfg, device,
            augment=True, verbose=False)

        model = CNNClassifier(cfg.num_classes).to(device)
        model.load_state_dict(best_state)
        test_acc = evaluate(model, test_loader, device)
        if verbose:
            print(f"[自训练 round {r}] val_acc={val_acc:.4f}  test_acc={test_acc:.4f}")

        rounds.append(dict(round=r, n_pseudo=int(len(sel_idx)),
                           pseudo_acc=diag_acc, val_acc=float(val_acc),
                           test_acc=float(test_acc), state=best_state))

        # 用本轮 CNN 给所有无标签样本重新打分，供下一轮“新增”高置信样本。
        if r < cfg.self_train_rounds:
            probs_u = predict_probs(model, X_all[unlabeled_idx], cfg, device)
            cnn_pseudo[unlabeled_idx] = probs_u.argmax(axis=1)
            cnn_conf[unlabeled_idx] = probs_u.max(axis=1)

    best = max(rounds, key=lambda d: d["val_acc"])      # 按 val 选轮
    return dict(rounds=rounds, best_round=best["round"],
                best_val=best["val_acc"], best_test=best["test_acc"],
                best_state=best["state"])
