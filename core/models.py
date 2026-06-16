"""网络结构：
  - CNNBackbone      共享卷积主干（自编码机与 SimCLR 复用，保证两条路线对比公平）
  - ConvAutoencoder  主干 + 对称解码器（重建）
  - SimCLRNet        主干 + 投影头（对比学习）
  - CNNClassifier    最终交付、用于在（伪）标签上训练的 CNN 分类器
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_act(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class CNNBackbone(nn.Module):
    """输入 (B,1,32,32) -> 表征向量 h (B, embed_dim)。

    自编码机和 SimCLR 共用同一主干，差异只在训练目标，便于公平对比两种表征。
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn_act(1, 32, stride=1),    # 32x32
            conv_bn_act(32, 64, stride=2),   # 16x16
            conv_bn_act(64, 128, stride=2),  # 8x8
            conv_bn_act(128, 256, stride=2), # 4x4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # -> (B,256,1,1)
        self.fc = nn.Linear(256, embed_dim)
        self.feat_channels = 256
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        z = self.pool(z).flatten(1)
        return self.fc(z)


class ConvDecoder(nn.Module):
    """从表征向量 h 重建 (B,1,32,32)。与主干大致对称。"""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 16x16
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 32x32
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),  # 输出像素 ∈ [0,1]，匹配 ToTensor 的取值范围
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.fc(h).view(-1, 256, 4, 4)
        return self.deconv(z)


class ConvAutoencoder(nn.Module):
    """卷积自编码机：encode 给嵌入，forward 给重建图。"""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.encoder = CNNBackbone(embed_dim)
        self.decoder = ConvDecoder(embed_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class ProjectionHead(nn.Module):
    """SimCLR 投影头：h -> z（仅训练时用于对比损失，下游用 h 不用 z）。"""

    def __init__(self, embed_dim: int = 128, proj_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class SimCLRNet(nn.Module):
    """SimCLR：主干 + 投影头。forward 返回归一化后的 z。"""

    def __init__(self, embed_dim: int = 128, proj_dim: int = 64):
        super().__init__()
        self.backbone = CNNBackbone(embed_dim)
        self.head = ProjectionHead(embed_dim, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        z = self.head(h)
        return F.normalize(z, dim=1)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """下游使用的表征 h（不经投影头）。"""
        return self.backbone(x)


class CNNClassifier(nn.Module):
    """最终交付的 CNN 分类器：(B,1,32,32) -> 10 类 logits。

    独立于表征主干，因为它是在（真+伪）标签上有监督训练的成品模型。
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                              # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                              # 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                              # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


if __name__ == "__main__":
    x = torch.randn(4, 1, 32, 32)
    ae = ConvAutoencoder(128)
    print("AE encode:", ae.encode(x).shape, "recon:", ae(x).shape)
    sim = SimCLRNet(128, 64)
    print("SimCLR z:", sim(x).shape, "h:", sim.encode(x).shape)
    clf = CNNClassifier()
    print("CLF logits:", clf(x).shape)
