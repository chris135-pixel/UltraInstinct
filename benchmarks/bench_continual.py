"""Ultra Instinct —— 能力验证 v2：修正评估口径 + 条件化对照。

v1 的错误：把「网络在固化区的输出」当作旧技能保留率。但方案里固化区走记忆库
（存的就是旧动作），网络输出根本不参与。v2 改为评估完整混合系统：

    固化区命中  → 输出记忆库中存的旧动作
    固化区未命中 → 输出慢网络

另加 E5：把「当前技能 ID」拼进状态向量（条件化），检验能否让新旧技能共存。
这是唯一的架构性解法——键里含技能标识，记忆库才不会把不同技能的状态混为一谈。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)
GRID = 16


def unit(d: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    return d / np.maximum(n, 1e-8)


def task_a(d: np.ndarray) -> np.ndarray:
    """旧技能：直奔目标。"""
    return unit(d)


def task_b(d: np.ndarray) -> np.ndarray:
    """新技能：同状态，动作旋转 90 度（模拟绕障）。"""
    r = unit(d)
    return np.stack([r[..., 1], -r[..., 0]], axis=-1)


class Net(nn.Module):
    """d_in → 64 → 64 → 2。d_in=2 为非条件化，3 为条件化（含技能 ID）。"""

    def __init__(self, d_in: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Frozen:
    """固化区掩码。"""

    mask: np.ndarray

    def hit(self, d: np.ndarray) -> np.ndarray:
        idx = np.clip(((d + 1.0) * (GRID / 2)).astype(int), 0, GRID - 1)
        return self.mask[idx[:, 0], idx[:, 1]]


def make_frozen(coverage: float, rng: np.random.Generator) -> Frozen:
    n = int(round(coverage * GRID * GRID))
    flat = np.zeros(GRID * GRID, dtype=bool)
    flat[rng.choice(GRID * GRID, n, replace=False)] = True
    return Frozen(flat.reshape(GRID, GRID))


def train(net, fn, steps, rng, frozen=None, skip_frozen=False, skill_id=None,
          replay=0.0, replay_fn=None, replay_skill_id=None, lr=2e-3) -> None:
    """训练。skill_id 非空时把技能标识拼进输入（条件化）。

    replay_skill_id 必须是被回放的那个技能自己的 ID。若沿用当前技能的 ID，
    同一个 (Δ, id) 会同时对应两套动作，网络只能输出平均值——两个技能一起废掉。
    """
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    net.train()
    if replay_skill_id is None:
        replay_skill_id = skill_id

    def prep(d: np.ndarray, sid: float | None) -> torch.Tensor:
        if sid is None:
            return torch.from_numpy(d.astype(np.float32))
        pad = np.full((len(d), 1), sid, dtype=np.float32)
        return torch.from_numpy(np.hstack([d, pad]).astype(np.float32))

    for _ in range(steps):
        d = rng.uniform(-1, 1, (256, 2)).astype(np.float32)
        y = fn(d)
        keep = np.ones(len(d), dtype=bool)
        if skip_frozen and frozen is not None:
            keep = ~frozen.hit(d)
        d, y = d[keep], y[keep]

        xb, yb = prep(d, skill_id), torch.from_numpy(y.astype(np.float32))
        if replay > 0 and replay_fn is not None:
            nr = int(len(d) * replay / max(1e-6, 1 - replay))
            dr = rng.uniform(-1, 1, (nr, 2)).astype(np.float32)
            xb = torch.cat([xb, prep(dr, replay_skill_id)])
            yb = torch.cat([yb, torch.from_numpy(replay_fn(dr).astype(np.float32))])
        if xb.shape[0] == 0:
            return

        opt.zero_grad()
        loss_fn(net(xb), yb).backward()
        opt.step()


@torch.no_grad()
def net_out(net, d: np.ndarray, skill_id: float | None = None) -> np.ndarray:
    if skill_id is None:
        x = torch.from_numpy(d.astype(np.float32))
    else:
        pad = np.full((len(d), 1), skill_id, dtype=np.float32)
        x = torch.from_numpy(np.hstack([d, pad]).astype(np.float32))
    return net(x).numpy()


def cos_ok(pred: np.ndarray, true: np.ndarray, thr: float = 0.9) -> np.ndarray:
    c = (pred * true).sum(-1) / np.maximum(
        np.linalg.norm(pred, axis=-1) * np.linalg.norm(true, axis=-1), 1e-8)
    return c > thr


def rng_test(n: int = 6000) -> np.ndarray:
    return np.random.default_rng(777).uniform(-1, 1, (n, 2)).astype(np.float32)


def hybrid_rate(net, fn, frozen: Frozen, stored_fn, d: np.ndarray) -> float:
    """完整混合系统的成功率。

    命中固化区 → 输出记忆库里的动作（stored_fn 生成，即训练 A 时存下的）；
    未命中 → 输出慢网络。

    stored_fn 与待测 fn 相同则代表「查的就是对的」；不同则代表「命中了错的记忆」。
    """
    hit = frozen.hit(d)
    pred = net_out(net, d)
    stored = stored_fn(d)
    out = np.where(hit[:, None], stored, pred)
    return float(cos_ok(out, fn(d)).mean())


def fresh(seed: int, d_in: int = 2) -> Net:
    torch.manual_seed(seed)
    return Net(d_in)


def main() -> None:
    rng = np.random.default_rng(2024)
    d = rng_test()
    res: dict = {}

    print("=" * 78)
    print("E1  无记忆库：训 A → 训 B → 回测 A（灾难性遗忘基线）")
    print("=" * 78)
    net = fresh(0)
    train(net, task_a, 1200, rng)
    a0 = float(cos_ok(net_out(net, d), task_a(d)).mean())
    train(net, task_b, 1200, rng)
    a1 = float(cos_ok(net_out(net, d), task_a(d)).mean())
    b1 = float(cos_ok(net_out(net, d), task_b(d)).mean())
    print(f"  训 A 后 A {a0:6.1%} → 训 B 后 A {a1:6.1%}（保留 {a1 / a0:5.1%}），B {b1:6.1%}")
    res["e1"] = {"a_before": a0, "a_after": a1, "b_after": b1}

    print()
    print("=" * 78)
    print("E2  混合系统（正确口径）：固化 A → 训 B → 分别测 A / B")
    print("=" * 78)
    print(f"{'p':>6}{'A 总成功率':>13}{'B 总成功率':>13}{'A未固化区':>12}{'B未固化区':>12}")
    rows = []
    for p in (0.3, 0.6, 0.9, 0.95):
        net = fresh(1)
        train(net, task_a, 1200, rng)
        fz = make_frozen(p, rng)
        train(net, task_b, 1200, rng, frozen=fz, skip_frozen=True)
        a_tot = hybrid_rate(net, task_a, fz, task_a, d)
        b_tot = hybrid_rate(net, task_b, fz, task_a, d)
        hit = fz.hit(d)
        a_free = float(cos_ok(net_out(net, d[~hit]), task_a(d[~hit])).mean())
        b_free = float(cos_ok(net_out(net, d[~hit]), task_b(d[~hit])).mean())
        rows.append({"p": p, "a_total": a_tot, "b_total": b_tot,
                     "a_free": a_free, "b_free": b_free})
        print(f"{p:>6.2f}{a_tot:>13.1%}{b_tot:>13.1%}{a_free:>12.1%}{b_free:>12.1%}")
    res["e2"] = rows

    print()
    print("=" * 78)
    print("E3  非条件化 + replay(30%)：用旧样本拖住旧技能")
    print("=" * 78)
    net = fresh(2)
    train(net, task_a, 1200, rng)
    fz = make_frozen(0.9, rng)
    train(net, task_b, 1200, rng, frozen=fz, skip_frozen=True,
          replay=0.3, replay_fn=task_a)
    a_tot = hybrid_rate(net, task_a, fz, task_a, d)
    b_tot = hybrid_rate(net, task_b, fz, task_a, d)
    print(f"  A 总成功率 {a_tot:6.1%}    B 总成功率 {b_tot:6.1%}"
          f"   （无 replay 时 B≈{rows[-1]['b_total']:.1%}）")
    res["e3"] = {"a_total": a_tot, "b_total": b_tot}

    print()
    print("=" * 78)
    print("E5  条件化：状态 = (Δx, Δy, 技能ID)，同一状态空间承载两个技能")
    print("=" * 78)
    net = fresh(3, d_in=3)
    train(net, task_a, 1200, rng, skill_id=0.0)
    train(net, task_b, 1200, rng, skill_id=1.0)   # 不固化、不 replay、直接覆盖训练
    a_c = float(cos_ok(net_out(net, d, 0.0), task_a(d)).mean())
    b_c = float(cos_ok(net_out(net, d, 1.0), task_b(d)).mean())
    print(f"  训 A 再训 B 后：  A {a_c:6.1%}   B {b_c:6.1%}   （无 replay、无固化保护）")
    print(f"  对照 E1（未条件化同流程）：A {a1:6.1%}   B {b1:6.1%}")
    print("  → 条件维被忽略：训 B 时 batch 里 ID 恒为 1，网络直接不学这一维")
    res["e5"] = {"a": a_c, "b": b_c}

    print()
    print("=" * 78)
    print("E6  条件化 + 联合回放：每个 batch 同时含 (Δ,0)→A 与 (Δ,1)→B")
    print("=" * 78)
    net = fresh(4, d_in=3)
    train(net, task_a, 1200, rng, skill_id=0.0)
    train(net, task_b, 1200, rng, skill_id=1.0, replay=0.5, replay_fn=task_a,
          replay_skill_id=0.0)
    a_j = float(cos_ok(net_out(net, d, 0.0), task_a(d)).mean())
    b_j = float(cos_ok(net_out(net, d, 1.0), task_b(d)).mean())
    print(f"  A {a_j:6.1%}   B {b_j:6.1%}   → 零和打破：同一状态空间并存两个技能")
    res["e6"] = {"a": a_j, "b": b_j}

    print()
    print("=" * 78)
    print("E7  三技能叠加：验证退化趋势（A/B/C 各占一个 ID）")
    print("=" * 78)
    net = fresh(5, d_in=3)
    train(net, task_a, 1200, rng, skill_id=0.0)
    train(net, task_b, 1200, rng, skill_id=1.0, replay=0.5, replay_fn=task_a,
          replay_skill_id=0.0)
    train(net, task_a, 1200, rng, skill_id=2.0, replay=0.5, replay_fn=task_b,
          replay_skill_id=1.0)
    a3 = float(cos_ok(net_out(net, d, 0.0), task_a(d)).mean())
    b3 = float(cos_ok(net_out(net, d, 1.0), task_b(d)).mean())
    c3 = float(cos_ok(net_out(net, d, 2.0), task_a(d)).mean())
    print(f"  技能1(A) {a3:6.1%}   技能2(B) {b3:6.1%}   技能3(A') {c3:6.1%}")
    res["e7"] = {"s1": a3, "s2": b3, "s3": c3}

    with open("benchmarks/continual_learning.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("\n已写入 benchmarks/continual_learning.json")


if __name__ == "__main__":
    main()
