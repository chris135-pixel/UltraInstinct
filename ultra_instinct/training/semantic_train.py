"""语义编码器训练：用部署方给的 KV 映射做监督对比学习。

核心思路（为什么这样训）：
    用户给的 KV 映射里，同一个答案 V 往往有多条问法 K（「它是什么狗」/
    「它是什么颜色的狗」→ 同一答案）。把这些同义问法的向量互相拉近，
    不同答案的推远，网络就学到「哪些 n-gram 组合在这个任务里等价」。

    这就是可迁移的边界：网络只学训练数据里出现过的等价关系。没出现过的
    说法能否命中，取决于它与训练样本的字面重叠——不是凭空理解语义。

训练目标 = 监督对比损失（Supervised Contrastive）：
    对每个样本 i，正样本 = 同一答案的其他问法，负样本 = 其余全部；
    loss_i = -mean_pos log( exp(sim/τ) / sum_{k≠i} exp(sim/τ) )

导出：
    训练用 torch，导出成 net.py 的 MLP 格式 npz（w1/b1/w2/b2/w3/b3/out_tanh），
    推理端 SemanticEncoder 用 numpy 前向，零 torch 依赖。

跑法：
    python -m ultra_instinct.training.semantic_train \
        --pairs examples/kv_example.json --out checkpoints/semantic.npz
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultra_instinct.encoding import ContentEncoder

# KV 映射格式：{答案: [问法1, 问法2, ...]}
Pairs = dict[str, list[str]]


class TorchSemanticNet(nn.Module):
    """与 net.py 的 MLP 结构逐层对应：Linear-Tanh-Linear-Tanh-Linear。"""

    def __init__(self, input_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def load_pairs(path: str) -> Pairs:
    """读 KV 映射 JSON：值为字符串列表（同一答案的多种问法）。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("KV 映射必须是 {答案: [问法, ...]} 形式的 JSON 对象")
    pairs: Pairs = {}
    for answer, variants in raw.items():
        if isinstance(variants, str):
            variants = [variants]
        if not isinstance(variants, list) or not all(
                isinstance(v, str) for v in variants):
            raise ValueError(f"答案 {answer!r} 的问法必须是字符串或字符串列表")
        if variants:
            pairs[str(answer)] = [str(v) for v in variants]
    if len(pairs) < 2:
        raise ValueError("至少需要 2 个不同答案才能做对比学习")
    return pairs


def flatten(pairs: Pairs) -> tuple[list[str], np.ndarray]:
    """摊平成 (文本列表, 标签数组)；标签相同的文本互为语义正样本。"""
    texts: list[str] = []
    labels: list[int] = []
    for label, (answer, variants) in enumerate(sorted(pairs.items())):
        # 答案本身也作为一条问法加入，让「标准说法」成为簇心之一
        for text in [answer, *variants]:
            texts.append(text)
            labels.append(label)
    return texts, np.asarray(labels, dtype=np.int64)


def supcon_loss(z: torch.Tensor, labels: torch.Tensor,
                tau: float = 0.1) -> torch.Tensor:
    """监督对比损失。z 已 L2 归一；同标签互为正样本。"""
    sim = (z @ z.T) / tau                       # (n, n)
    n = len(z)
    mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & \
               ~torch.eye(n, dtype=torch.bool, device=z.device)
    # 数值稳定：减每行最大
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim)
    denom = exp_sim.sum(dim=1) - exp_sim.diag()  # 排除自身
    has_pos = mask_pos.any(dim=1)
    if not has_pos.any():
        # 每个答案只有一条样本时无从构造正样本对
        raise ValueError("每个答案至少需要 2 条不同问法（含答案本身）才能训练")
    numer = (exp_sim * mask_pos).sum(dim=1)
    per_anchor = -torch.log((numer / denom).clamp_min(1e-9))
    return per_anchor[has_pos].mean()


def train_semantic(pairs: Pairs,
                   base: ContentEncoder | None = None,
                   steps: int = 800, hidden: int = 64, out_dim: int = 64,
                   lr: float = 3e-3, tau: float = 0.1,
                   batch: int = 0, seed: int = 0,
                   log: Callable[[str], None] | None = None) -> np.lib.npyio.NpzFile | dict:
    """训练语义网络，返回可直接喂 MLP 权重的 dict（w1/b1/.../out_tanh）。

    Args:
        pairs: {答案: [同义问法, ...]}，至少 2 个答案，每个答案至少 1 条问法。
        steps: 梯度步数。
        hidden / out_dim: 隐层与语义向量维度。
        lr / tau: 学习率、对比温度。
        batch: 批大小，0 表示全量（小数据集默认全量）。
        seed: 随机种子，保证可复现。
        log: 可选日志回调。

    Returns:
        dict: w1,b1,w2,b2,w3,b3 (numpy, 转置成 (in,out)) 与 out_tanh=False。
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    random.seed(seed)
    base = base or ContentEncoder(dim=256)
    texts, labels_np = flatten(pairs)

    # 每个答案需至少 2 条（答案本身 + 1 条问法）才有正样本对
    counts = np.bincount(labels_np)
    if counts.min() < 2:
        bad = sorted({t for t, l in zip(texts, labels_np)
                      if counts[l] < 2})
        raise ValueError(
            f"以下答案只有一条问法，无法构造正样本对：{bad}；"
            "请为每个答案提供至少一条同义问法")

    X = np.stack([base.features(t) for t in texts]).astype(np.float32)
    tx = torch.from_numpy(X)
    ty = torch.from_numpy(labels_np)

    net = TorchSemanticNet(input_dim=base.dim, hidden=hidden, out_dim=out_dim)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    batch = int(batch) if batch else len(texts)
    last = float("nan")
    net.train()
    for step in range(steps):
        idx = torch.from_numpy(rng.integers(0, len(texts), size=batch))
        xb, yb = tx[idx], ty[idx]
        # 一个批里每个标签至少要有 2 条，否则 SupCon 分母/分子退化
        if torch.bincount(yb).min() < 2:
            continue
        z = F.normalize(net(xb), dim=1)
        loss = supcon_loss(z, yb, tau=tau)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.item())
        if log is not None and (step % max(1, steps // 5) == 0 or step == steps - 1):
            log(f"  step {step:>4d}/{steps}  loss={last:.4f}")

    net.eval()
    # 导出成 net.py MLP 的格式：Linear.weight 是 (out,in)，转置成 (in,out)
    w = [layer.weight.detach().numpy().T.copy()
         for layer in (net.body[0], net.body[2], net.body[4])]
    b = [layer.bias.detach().numpy().copy()
         for layer in (net.body[0], net.body[2], net.body[4])]
    return {
        "w1": np.ascontiguousarray(w[0], dtype=np.float32),
        "b1": np.ascontiguousarray(b[0], dtype=np.float32),
        "w2": np.ascontiguousarray(w[1], dtype=np.float32),
        "b2": np.ascontiguousarray(b[1], dtype=np.float32),
        "w3": np.ascontiguousarray(w[2], dtype=np.float32),
        "b3": np.ascontiguousarray(b[2], dtype=np.float32),
        "out_tanh": np.bool_(False),
    }


def save_weights(weights: dict, path: str) -> None:
    """把训练结果写成 net.py MLP.load 能读的 npz。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(p), **weights)


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="训练语义编码器（KV 映射 → npz 权重）")
    ap.add_argument("--pairs", required=True, help="KV 映射 JSON：{答案: [同义问法, ...]}")
    ap.add_argument("--out", required=True, help="权重输出 npz 路径")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--out-dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=0, help="0=全量")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(list(argv) if argv is not None else None)

    pairs = load_pairs(args.pairs)
    print(f"载入 {len(pairs)} 个答案，"
          f"{sum(1 + len(v) for v in pairs.values())} 条训练文本")
    weights = train_semantic(
        pairs, steps=args.steps, hidden=args.hidden, out_dim=args.out_dim,
        lr=args.lr, tau=args.tau, batch=args.batch, seed=args.seed,
        log=print)
    save_weights(weights, args.out)
    print(f"权重已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
