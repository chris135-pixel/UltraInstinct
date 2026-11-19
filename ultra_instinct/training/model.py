"""训练用的网络定义与超参。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TrainConfig:
    """训练超参。

    Attributes:
        hidden: 隐层宽度。实测 64 与 16 的单步延迟只差 0.86 μs，
            但容量差 13 倍，所以不要为了速度缩模型。
        steps: 每个技能的梯度步数。
        lr: 学习率。
        batch: 批大小。
        replay_ratio: 训练新技能时，批里混入旧技能样本的比例。
            实测：不混则旧技能掉到 0%（E1），混 50% 后新旧各 100%（E6）。
        out_tanh: 输出层是否过 tanh。方向型动作用 True。
    """

    hidden: int = 64
    steps: int = 1200
    lr: float = 2e-3
    batch: int = 256
    replay_ratio: float = 0.5
    out_tanh: bool = True


class TorchMLP(nn.Module):
    """与推理端 MLP 结构严格对应的三层网络。"""

    def __init__(self, input_dim: int, hidden: int, output_dim: int,
                 out_tanh: bool = True) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.out_tanh = out_tanh
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.body(x)
        return torch.tanh(y) if self.out_tanh else y
