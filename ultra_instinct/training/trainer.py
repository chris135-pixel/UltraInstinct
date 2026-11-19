"""训练器：条件化 + 联合回放。

两个反直觉的实测结论决定了这里的实现：

1. 只把技能 ID 拼进输入不够（E5：旧技能剩 0.2%）。
   原因是训练新技能时，整个批里的 ID 维度恒定，网络会直接忽略这一维。
   必须让 ID 在同一个批里有变化 —— 也就是联合回放。

2. 回放样本必须带它自己所属的 skill_id（E6 第一版全 0 的教训）。
   如果回放样本沿用当前技能的 ID，同一个 (Δ, 1.0) 有时标 A 有时标 B，
   网络收到矛盾标签，两个技能一起废掉。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ultra_instinct.training.model import TrainConfig, TorchMLP


class ReplayBuffer:
    """存 (输入, 标签)，输入里已经包含技能标识，取出来直接用。"""

    def __init__(self, capacity: int = 200_000) -> None:
        self.capacity = capacity
        self._x: list[np.ndarray] = []
        self._y: list[np.ndarray] = []
        self._n = 0

    def __len__(self) -> int:
        return self._n

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        self._x.append(np.asarray(x, dtype=np.float32))
        self._y.append(np.asarray(y, dtype=np.float32))
        self._n += len(x)
        while self._n > self.capacity and len(self._x) > 1:
            self._n -= len(self._x.pop(0))
            self._y.pop(0)

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        x = np.concatenate(self._x)
        y = np.concatenate(self._y)
        idx = rng.integers(0, len(x), size=n)
        return x[idx], y[idx]


def train_skill(net: TorchMLP, task_fn, cfg: TrainConfig,
                rng: np.random.Generator, skill_id: int, n_skills: int,
                replay: ReplayBuffer | None = None,
                mask_fn=None, dim: int = 2) -> float:
    """训练一个技能。

    Args:
        net: 待训练网络（可以是已经训过别的技能的同一个网络）。
        task_fn: (n, dim) -> (n, output_dim)，理想动作。
        cfg: 超参。
        rng: numpy 随机源。
        skill_id: 当前技能下标。
        n_skills: 技能总数，用于归一化标识。
        replay: 旧样本缓冲区。为空则纯新技能训练（会忘掉旧技能）。
        mask_fn: (n, dim) -> (n,) bool，True 的样本丢弃。
            用于「只在不固化区学新动作」这类设置。

    Returns:
        最后一步的损失。
    """
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    net.train()

    code = 0.0 if n_skills <= 1 else skill_id / (n_skills - 1)
    loss_val = float("nan")

    for _ in range(cfg.steps):
        d = rng.uniform(-1, 1, (cfg.batch, dim)).astype(np.float32)
        if mask_fn is not None:
            d = d[~mask_fn(d)]
            if len(d) == 0:
                continue
        y = np.asarray(task_fn(d), dtype=np.float32)
        x = np.concatenate([d, np.full((len(d), 1), code, dtype=np.float32)], axis=1)

        if replay is not None and len(replay) > 0 and cfg.replay_ratio > 0:
            k = min(int(cfg.batch * cfg.replay_ratio),
                    max(1, cfg.batch - len(d)))
            if k > 0:
                xr, yr = replay.sample(k, rng)
                x = np.concatenate([x, xr])
                y = np.concatenate([y, yr])

        # 本批样本入库，带的都是当前技能的标识
        if replay is not None:
            replay.add(x[: len(d)], y[: len(d)])

        tx = torch.from_numpy(x).to(next(net.parameters()).dtype)
        ty = torch.from_numpy(y).to(next(net.parameters()).dtype)
        opt.zero_grad()
        loss = loss_fn(net(tx), ty)
        loss.backward()
        opt.step()
        loss_val = float(loss.item())

    net.eval()
    return loss_val
