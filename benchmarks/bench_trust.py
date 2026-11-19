"""Ultra Instinct —— 剔除工业流程约束后，重新量化「准确性 + 安全性」。

去掉认证/审批/变更管理之后，真正要回答的是三个纯技术问题：

    R1  固化区边界的动作突跳，靠算法能不能修平？代价是多少？
    R2  固化门槛放到 N=5，靠运行期自检兜底，净风险比 N=30 高多少？
    R3  准确性在闭环控制下由什么决定？单步量化误差重要吗？

R3 是前面几轮最容易搞错的地方：一直在用「开环单步角度误差」评价精度，
而真实系统是闭环的——单步偏 3 度，闭环会自己纠回来。
"""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)


# ------------------------------------------------------------------ 公共


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def unit(d: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    return d / np.maximum(n, 1e-8)


def field(d: np.ndarray) -> np.ndarray:
    """带曲率的动作场：中远距离绕行，近目标收敛为直行。

    曲率在近目标处必须衰减，否则任何实现（含理想解析解）都会绕圈，
    那样测出来的是任务设计问题，不是量化误差。
    """
    r = np.linalg.norm(d, axis=-1)
    ang = 1.05 * np.exp(-r / 0.35) / (1.0 + np.exp(-(r - 0.35) / 0.07))
    c, s = np.cos(ang), np.sin(ang)
    u = unit(d)
    return np.stack([u[..., 0] * c - u[..., 1] * s,
                     u[..., 0] * s + u[..., 1] * c], axis=-1)


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh(),
                                 nn.Linear(64, 2), nn.Tanh())

    def forward(self, x):
        return self.net(x)


def train(net, fn, steps, rng, mask=None, lr=2e-3) -> None:
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    net.train()
    for _ in range(steps):
        d = rng.uniform(-1, 1, (256, 2)).astype(np.float32)
        if mask is not None:
            d = d[~mask(d)]
            if len(d) == 0:
                continue
        y = fn(d).astype(np.float32)
        opt.zero_grad()
        loss_fn(net(torch.from_numpy(d)), torch.from_numpy(y)).backward()
        opt.step()


# ------------------------------------------------------------------ R1 边界连续性


def build_lut(grid: int, skill: str = "a") -> np.ndarray:
    """把技能刻进网格：返回 (grid, grid, 2) 的动作表。"""
    ax = (np.arange(grid) + 0.5) / grid * 2 - 1
    gx, gy = np.meshgrid(ax, ax, indexing="ij")
    centers = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    act = field(centers) if skill == "a" else np.stack(
        [unit(centers)[:, 1], -unit(centers)[:, 0]], axis=-1)
    return act.reshape(grid, grid, 2)


def r1_boundary(grid: int = 16, frozen_ratio: float = 0.9, seed: int = 42) -> dict:
    """沿一条穿越固化区边界的直线采样，测相邻采样点动作的角度差。

    四种输出策略：
      hard     命中查表 A，未命中走网络（网络已学 B）—— 二值切换
      alpha    对固化掩码做双线性插值当权重，边界处连续换权
      clamp    alpha + 单步动作变化限幅
    """
    rng = np.random.default_rng(seed)
    frozen_flat = np.zeros(grid * grid, dtype=bool)
    frozen_flat[rng.choice(grid * grid,
                           int(frozen_ratio * grid * grid), replace=False)] = True
    frozen = frozen_flat.reshape(grid, grid)
    lut_a = build_lut(grid, "a")

    def to_idx(d):
        return np.clip(((d + 1.0) * grid / 2).astype(int), 0, grid - 1)

    def mask_fn(d):
        idx = to_idx(d)
        return frozen[idx[:, 0], idx[:, 1]]

    torch.manual_seed(seed)
    net = Net()
    train(net, lambda d: field(d), 1200, rng)
    train(net, lambda d: np.stack([unit(d)[:, 1], -unit(d)[:, 0]], axis=-1),
          1200, rng, mask=mask_fn)

    # 采样：三条水平扫描线，刻意避开原点区域。
    # field 在近原点处方向本身翻转 90°（角向奇异），那是任务定义的问题，
    # 闭环里 Δ→0 就停，不会碰到；混进来会污染「切换缺陷」的度量。
    lines = [np.stack([np.linspace(-0.9, 0.9, 2200), np.full(2200, y)], axis=-1)
             for y in (0.45, 0.60, 0.75)]

    def continuity(acts: np.ndarray) -> dict:
        a = unit(acts[:-1])
        b = unit(acts[1:])
        ang = np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))
        return {
            "mean_deg": float(ang.mean()),
            "p95_deg": float(np.percentile(ang, 95)),
            "max_deg": float(ang.max()),
            "over_30_deg": float((ang > 30).mean()),
        }

    per_line: dict[str, list[dict]] = {}
    cross: dict[str, list[float]] = {}
    net_calls = {"hard": [], "alpha": []}

    for path in lines:
        with torch.no_grad():
            net_out = net(torch.from_numpy(path.astype(np.float32))).numpy()
        idx = to_idx(path)
        hit = frozen[idx[:, 0], idx[:, 1]]
        lut_out = lut_a[idx[:, 0], idx[:, 1]]

        # 权重场：对 0/1 掩码做插值，边界处自然得到 0~1 的过渡，不再突变。
        # 注意不能用「到最近格中心的距离」——那个量在格边界上是不连续的。
        alpha = np.clip(lut_query(frozen.astype(np.float64), path, True), 0.0, 1.0)

        variants = {
            # 硬切换：命中即 A，否则网络（网络的 B 输出）
            "hard": np.where(hit[:, None], lut_out, net_out),
            # 连续混合：α 由掩码插值决定，边界处两路逐渐换权
            "alpha": alpha[:, None] * lut_out + (1 - alpha[:, None]) * net_out,
        }
        # 限幅：对 alpha 版逐点施加单步变化上限 8°
        seq = variants["alpha"].copy()
        for i in range(1, len(seq)):
            a0, a1 = unit(seq[i - 1]), unit(seq[i])
            ang = np.degrees(np.arccos(float(np.clip((a0 * a1).sum(), -1, 1))))
            if ang > 8.0:
                axis = np.cross(np.append(a0, 0), np.append(a1, 0))[2]
                s = np.sign(axis) if abs(axis) > 1e-9 else 1.0
                th = np.radians(8.0) * s
                c, sn = np.cos(th), np.sin(th)
                seq[i] = np.array([a0[0] * c - a0[1] * sn, a0[0] * sn + a0[1] * c])
        variants["clamp"] = seq

        for k, v in variants.items():
            per_line.setdefault(k, []).append(continuity(v))

        # 只统计「命中/未命中翻转」的那一对相邻采样点——缺陷真正发生的位置
        edge = np.where(hit[:-1] != hit[1:])[0]
        for k in ("hard", "alpha"):
            acts = variants[k]
            if len(edge):
                a = unit(acts[edge])
                b = unit(acts[edge + 1])
                cross.setdefault(k, []).extend(
                    np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1))).tolist())

        net_calls["hard"].append(float((~hit).mean()))
        net_calls["alpha"].append(float((alpha < 0.999).mean()))

    out = {}
    for k, rows in per_line.items():
        out[k] = {m: float(np.mean([r[m] for r in rows])) for m in rows[0]}
        out[k]["max_deg"] = float(np.max([r["max_deg"] for r in rows]))
    cross_stats = {}
    for k, v in cross.items():
        v = np.asarray(v)
        cross_stats[k] = {
            "n": int(len(v)),
            "mean_deg": float(v.mean()),
            "max_deg": float(v.max()),
            "over_30_deg": float((v > 30).mean()),
        }
    out["net_call_ratio"] = {k: float(np.mean(v)) for k, v in net_calls.items()}
    out["crossing"] = cross_stats
    return out


# ------------------------------------------------------------------ R2 固化门槛 vs 自检


def r2_threshold(T: int = 100, samples: int = 200_000, seed: int = 7) -> dict:
    """N 次全成功 → 后验 Beta(N+1,1)。统计条目被使用 T 次时的漏检失败次数。

    自检机制：连续 m 次无进展就回退到网络。失败次数被截断在 m。
    """
    rng = np.random.default_rng(seed)
    rows = []
    for n in (3, 5, 10, 20, 30, 50):
        p = rng.beta(n + 1, 1.0, samples)          # 后验真实成功率
        fails = T * (1.0 - p)
        row = {"n": n, "post_mean": float(p.mean())}
        for m in (None, 3, 2, 1):
            capped = fails if m is None else np.minimum(fails, m)
            row[f"fail_m{m}"] = float(capped.mean())
        rows.append(row)

    # 练习成本：一个状态簇被访问 k 次的概率，决定能否凑够 N 次全成功
    lam = 8.0
    from math import exp, factorial
    cover = {}
    for n in (3, 5, 10, 20, 30, 50):
        # P(k >= N)，k~Poisson(lam)
        pk = 1.0 - sum(exp(-lam) * lam ** k / factorial(k) for k in range(n))
        cover[n] = max(pk, 0.0)
    return {"table": rows, "coverage_lambda8": cover}


# ------------------------------------------------------------------ R3 闭环精度


def lut_query(lut: np.ndarray, d: np.ndarray, bilinear: bool) -> np.ndarray:
    """查表。bilinear=True 时用四邻域插值，否则取最近格中心。"""
    grid = lut.shape[0]
    x = (d + 1.0) * grid / 2 - 0.5
    x = np.clip(x, 0.0, grid - 1.0)   # 先 clamp 坐标，否则 f 会越界成负权重
    if not bilinear:
        i = np.clip(np.round(x).astype(int), 0, grid - 1)
        return lut[i[:, 0], i[:, 1]]
    x0 = np.floor(x).astype(int)
    f = x - x0
    x0 = np.clip(x0, 0, grid - 2)
    a = lut[x0[:, 0], x0[:, 1]]
    b = lut[x0[:, 0] + 1, x0[:, 1]]
    c = lut[x0[:, 0], x0[:, 1] + 1]
    e = lut[x0[:, 0] + 1, x0[:, 1] + 1]
    # 标量场（掩码）时 a 是 (n,)，向量场（动作）时是 (n,2)，统一广播
    shp = (-1,) + (1,) * (a.ndim - 1)
    fx, fy = f[:, 0].reshape(shp), f[:, 1].reshape(shp)
    return (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy)
            + c * (1 - fx) * fy + e * fx * fy)


def r3_closed_loop(n_ep: int = 400, step: float = 0.03,
                   tol: float = 0.01, max_steps: int = 150,
                   seed: int = 11) -> dict:
    """闭环到达：起点在半径 0.6~0.95，目标在半径 0.05 内。

    测的是「最终到达误差 + 路径里程」，不是单步开环角度误差。
    """
    rng = np.random.default_rng(seed)
    th0 = rng.uniform(0, 2 * np.pi, n_ep)
    r0 = rng.uniform(0.6, 0.95, n_ep)
    pos = np.stack([r0 * np.cos(th0), r0 * np.sin(th0)], axis=-1)
    goal = np.stack([rng.uniform(-0.05, 0.05, n_ep),
                     rng.uniform(-0.05, 0.05, n_ep)], axis=-1)

    luts = {g: build_lut(g, "a") for g in (8, 16, 32, 64, 128)}

    torch.manual_seed(seed)
    net = Net()
    train(net, lambda d: field(d), 3000, rng)

    def run(policy) -> dict:
        p = pos.copy()
        dist = np.zeros(n_ep)
        alive = np.ones(n_ep, dtype=bool)
        for _ in range(max_steps):
            d = goal - p
            if not alive.any():
                break
            a = policy(d)
            nrm = np.linalg.norm(d, axis=-1)
            # 末段步长随距离衰减：否则固定步长会跨过容差来回振荡，
            # 测出来的是采样问题而不是精度问题
            st = np.minimum(step, np.maximum(nrm, 1e-9))
            nxt = p + st[:, None] * unit(a) * alive[:, None]
            dist += np.linalg.norm(nxt - p, axis=-1) * alive
            p = nxt
            alive &= np.linalg.norm(goal - p, axis=-1) > tol
        final = np.linalg.norm(goal - p, axis=-1)
        straight = np.linalg.norm(pos - goal, axis=-1)
        return {
            "final_err_mm": float(final.mean() * 1000),
            "final_p95_mm": float(np.percentile(final, 95) * 1000),
            "reached": float((final <= tol).mean()),
            "path_ratio": float((dist / straight).mean()),
        }

    res = {"network": run(lambda d: net(torch.from_numpy(
        d.astype(np.float32))).detach().numpy())}
    for g, lut in luts.items():
        res[f"lut{g}_nearest"] = run(lambda d, l=lut: lut_query(l, d, False))
        res[f"lut{g}_bilinear"] = run(lambda d, l=lut: lut_query(l, d, True))
    return res


# ------------------------------------------------------------------ 主流程


def main() -> None:
    print("=" * 80)
    print("R1  固化区边界：相邻采样点动作角度差（间距 0.0008，格宽 0.125）")
    print("=" * 80)
    r1 = r1_boundary()
    print("  全路径（含区内部，被大量平滑点稀释，只看 max 有意义）")
    print(f"{'策略':>8}{'平均':>9}{'p95':>9}{'最大':>9}{'>30°占比':>11}")
    for k in ("hard", "alpha", "clamp"):
        v = r1[k]
        print(f"{k:>8}{v['mean_deg']:>8.2f}°{v['p95_deg']:>8.2f}°"
              f"{v['max_deg']:>8.2f}°{v['over_30_deg']:>10.1%}")
    print("  仅跨边界点对（缺陷真正发生处）")
    print(f"{'策略':>8}{'样本数':>9}{'平均':>9}{'最大':>9}{'>30°占比':>11}")
    for k, v in r1["crossing"].items():
        print(f"{k:>8}{v['n']:>9}{v['mean_deg']:>8.2f}°"
              f"{v['max_deg']:>8.2f}°{v['over_30_deg']:>10.1%}")
    print(f"  网络调用率：hard {r1['net_call_ratio']['hard']:.1%}  "
          f"alpha {r1['net_call_ratio']['alpha']:.1%}")

    print()
    print("=" * 80)
    print("R2  固化门槛 N vs 运行期自检（条目使用 100 次，失败次数期望）")
    print("=" * 80)
    r2 = r2_threshold()
    print(f"{'N':>5}{'后验均值':>10}{'无自检':>10}{'m=3':>9}{'m=2':>9}{'m=1':>9}"
          f"{'覆盖率':>10}")
    for row in r2["table"]:
        cov = r2["coverage_lambda8"][row["n"]]
        print(f"{row['n']:>5}{row['post_mean']:>9.1%}{row['fail_mNone']:>10.1f}"
              f"{row['fail_m3']:>9.2f}{row['fail_m2']:>9.2f}{row['fail_m1']:>9.2f}"
              f"{cov:>10.1%}")

    print()
    print("=" * 80)
    print("R3  闭环到达精度（容差 10 mm，单步 30 mm，末段步长随距离衰减）")
    print("=" * 80)
    r3 = r3_closed_loop()
    print(f"{'策略':>18}{'最终误差':>11}{'p95':>10}{'到达率':>9}{'路径/直线':>11}")
    for k, v in r3.items():
        print(f"{k:>18}{v['final_err_mm']:>10.1f}mm{v['final_p95_mm']:>9.1f}mm"
              f"{v['reached']:>9.1%}{v['path_ratio']:>11.3f}")

    with open("benchmarks/trust_accuracy.json", "w", encoding="utf-8") as f:
        json.dump({"r1_boundary": r1, "r2_threshold": r2, "r3_closed_loop": r3},
                  f, ensure_ascii=False, indent=2)
    print("\n已写入 benchmarks/trust_accuracy.json")


if __name__ == "__main__":
    main()
