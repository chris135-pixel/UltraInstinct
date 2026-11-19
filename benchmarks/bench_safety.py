"""Ultra Instinct —— 安全性与准确性审查（不追求速度，量化风险）。

三个必须回答的问题：
    S1  离散量化会不会让动作精度崩掉？128x128 网格够不够？
    S2  固化判据 N=5 次的统计置信有多低？
    S3  固化区与未固化区的边界上，输出动作会不会突跳？

S3 是混合架构最危险的缺陷：边界一侧查表得到旧技能动作，另一侧走网络得到
新技能动作，两者可能相差 90 度。真实机械系统上这就是一次冲击。
"""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)


# ---------------------------------------------------------------- S1 量化误差


def action_field(d: np.ndarray) -> np.ndarray:
    """带曲率的动作场：靠近目标时大幅绕行，模拟避障。

    方向随位置变化快，这是量化误差最容易暴露的场景——比线性方向场严苛得多。
    """
    r = np.linalg.norm(d, axis=-1)
    ang = 1.05 * np.exp(-r / 0.3)          # 近目标最大偏转约 60°
    c, s = np.cos(ang), np.sin(ang)
    u = d / np.maximum(r[..., None], 1e-8)
    return np.stack([u[..., 0] * c - u[..., 1] * s,
                     u[..., 0] * s + u[..., 1] * c], axis=-1)


def quantize_error(grid: int, n: int = 40_000) -> dict[str, float]:
    """格子存格中心动作，查询点是格内均匀随机点，测量角度误差。"""
    rng = np.random.default_rng(3)
    d = rng.uniform(-1, 1, (n, 2))

    # 查询点所属格中心
    idx = np.clip(((d + 1.0) * grid / 2).astype(int), 0, grid - 1)
    center = (idx + 0.5) * (2.0 / grid) - 1.0

    pred = action_field(center)            # 查表得到的动作
    true = action_field(d)                 # 真实该执行的动
    cos = (pred * true).sum(-1)
    ang_err = np.degrees(np.arccos(np.clip(cos, -1, 1)))

    return {
        "grid": grid,
        "cell_size": 2.0 / grid,
        "mean_deg": float(ang_err.mean()),
        "p95_deg": float(np.percentile(ang_err, 95)),
        "p99_deg": float(np.percentile(ang_err, 99)),
        "bad_ratio": float((ang_err > 15).mean()),  # 误差超 15 度的比例
    }


# ---------------------------------------------------------------- S2 固化判据置信度


def confidence_lower(n_success: int) -> float:
    """n 次全成功时，真实成功率的 95% 置信下界（Beta(n+1,1) 的 5% 分位）。

    后验 CDF(x) = x^(n+1)，解 x^(n+1)=0.05 即得下界。
    """
    return float(0.05 ** (1.0 / (n_success + 1)))


# ---------------------------------------------------------------- S3 边界跳变


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh(),
                                 nn.Linear(64, 2), nn.Tanh())

    def forward(self, x):
        return self.net(x)


def unit(d: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    return d / np.maximum(n, 1e-8)


def task_a(d): return unit(d)


def task_b(d):
    r = unit(d)
    return np.stack([r[..., 1], -r[..., 0]], axis=-1)


def train(net, fn, steps, rng, mask=None, lr=2e-3):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    net.train()
    for _ in range(steps):
        d = rng.uniform(-1, 1, (256, 2)).astype(np.float32)
        if mask is not None:
            keep = ~mask(d)
            d = d[keep]
            if len(d) == 0:
                return
        y = fn(d).astype(np.float32)
        opt.zero_grad()
        loss_fn(net(torch.from_numpy(d)), torch.from_numpy(y)).backward()
        opt.step()


def boundary_jump(grid: int = 16, frozen_ratio: float = 0.9, seed: int = 42) -> dict[str, float]:
    """在固化区边界处，测量相邻两侧输出动作的角度差。

    固化区查表得到旧技能 A 的动作，未固化区走网络得到新技能 B 的动作。
    """
    rng = np.random.default_rng(seed)
    frozen_flat = np.zeros(grid * grid, dtype=bool)
    n_frozen = int(frozen_ratio * grid * grid)
    frozen_flat[rng.choice(grid * grid, n_frozen, replace=False)] = True
    frozen = frozen_flat.reshape(grid, grid)

    def mask_fn(d: np.ndarray) -> np.ndarray:
        idx = np.clip(((d + 1.0) * grid / 2).astype(int), 0, grid - 1)
        return frozen[idx[:, 0], idx[:, 1]]

    torch.manual_seed(seed)
    net = Net()
    train(net, task_a, 1200, rng)
    train(net, task_b, 1200, rng, mask=mask_fn)  # 未固化区学新技能 B

    # 收集「命中格」与「未命中格」相邻的格子对
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(grid):
        for j in range(grid):
            if not frozen[i, j]:
                continue
            for di, dj in ((1, 0), (0, 1)):
                ni, nj = i + di, j + dj
                if ni < grid and nj < grid and not frozen[ni, nj]:
                    ci = np.array([[(i + 0.5), (j + 0.5)]]) / grid * 2 - 1
                    cj = np.array([[(ni + 0.5), (nj + 0.5)]]) / grid * 2 - 1
                    pairs.append((ci, cj))
    if not pairs:
        return {}

    with torch.no_grad():
        a = np.concatenate([p[0] for p in pairs])
        b = np.concatenate([p[1] for p in pairs])
        out_a = task_a(a)                       # 固化区：查表 → 旧技能动作
        out_b = net(torch.from_numpy(b.astype(np.float32))).numpy()

    cos = (out_a * out_b).sum(-1) / np.maximum(
        np.linalg.norm(out_a, axis=-1) * np.linalg.norm(out_b, axis=-1), 1e-8)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return {
        "n_boundary_pairs": len(pairs),
        "mean_deg": float(ang.mean()),
        "p95_deg": float(np.percentile(ang, 95)),
        "max_deg": float(ang.max()),
        "over_30_deg_ratio": float((ang > 30).mean()),
    }


# ---------------------------------------------------------------- 主流程


def main() -> None:
    print("=" * 78)
    print("S1  量化误差：网格分辨率 → 动作角度误差（带曲率动作场）")
    print("=" * 78)
    print(f"{'grid':>6}{'格宽':>9}{'平均误差':>11}{'p95':>9}{'p99':>9}{'>15° 占比':>11}")
    s1 = []
    for g in (16, 32, 64, 128, 256, 512):
        r = quantize_error(g)
        s1.append(r)
        print(f"{g:>6}{r['cell_size']:>9.4f}{r['mean_deg']:>10.2f}°"
              f"{r['p95_deg']:>8.2f}°{r['p99_deg']:>8.2f}°{r['bad_ratio']:>10.1%}")

    print()
    print("=" * 78)
    print("S2  固化判据：N 次连续成功 → 真实成功率的 95% 置信下界")
    print("=" * 78)
    print(f"{'N':>6}{'置信下界':>12}{'结论':>28}")
    s2 = []
    for n in (3, 5, 10, 20, 30, 50, 100):
        lo = confidence_lower(n)
        verdict = "不可接受" if lo < 0.8 else ("勉强" if lo < 0.9 else "可用")
        s2.append({"n": n, "lower": lo})
        print(f"{n:>6}{lo:>11.1%}{verdict:>28}")

    print()
    print("=" * 78)
    print("S3  固化区边界跳变：命中侧(旧技能) vs 未命中侧(新技能)")
    print("=" * 78)
    s3 = boundary_jump()
    if s3:
        print(f"  采样边界对 {s3['n_boundary_pairs']} 组")
        print(f"  相邻格动作角度差：平均 {s3['mean_deg']:.1f}°  "
              f"p95 {s3['p95_deg']:.1f}°  最大 {s3['max_deg']:.1f}°")
        print(f"  超过 30° 的边界占比：{s3['over_30_deg_ratio']:.1%}")

    with open("benchmarks/safety_accuracy.json", "w", encoding="utf-8") as f:
        json.dump({"s1_quantization": s1, "s2_confidence": s2, "s3_boundary": s3},
                  f, ensure_ascii=False, indent=2)
    print("\n已写入 benchmarks/safety_accuracy.json")


if __name__ == "__main__":
    main()
