"""分级记忆实验：压缩损失、可压性、再巩固回填、体积-精度前沿。

跑法：
    python benchmarks/bench_hierarchy.py

四个实验，每个都对应一个可证伪的预测：

H1  压缩损失应等于曲率（二阶项）。块越大误差越大，且均值模型远差于仿射模型。
H2  不加 coherence 门槛直接合并，在**多值动作场**上应出现灾难性错误（>90°）。
    平滑场上门槛不触发 —— 这本身是结论：门槛是零成本的保险，不是常用件。
H3  再巩固回填：有要旨先验时误差起点低，且随着再遇到次数逼近逐字水平。
H4  体积-精度前沿：同样的占用体积，块均值 / 仿射 / 逐字三者怎么排。

评估集一律来自**真实到达轨迹**，不是均匀撒点：
均匀撒点会把大量「从没练过」的状态算进来，掩盖掉压缩本身的差异。
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultra_instinct.hierarchy import Hierarchy, compress, query_gists
from ultra_instinct.memory import LUTMemory
from ultra_instinct.quantize import Grid

GRID = 32
EXTENT = 1.0
STEP = 0.03
TOL = 0.02
N_PRACTICE = 420


def field(d: np.ndarray) -> np.ndarray:
    """平滑动作场：中远距离绕行，近目标收敛直行（与 examples/reach_demo 同任务）。"""
    d = np.atleast_2d(np.asarray(d, dtype=np.float64))
    r = np.linalg.norm(d, axis=-1)
    ang = 1.05 * np.exp(-r / 0.35) / (1.0 + np.exp(-(r - 0.35) / 0.07))
    c, s = np.cos(ang), np.sin(ang)
    u = d / np.maximum(r[..., None], 1e-8)
    return np.stack([u[..., 0] * c - u[..., 1] * s,
                     u[..., 0] * s + u[..., 1] * c], axis=-1).astype(np.float32)


def field_seam(d: np.ndarray) -> np.ndarray:
    """多值动作场：中线两侧各有一条合法通路（绕障可左可右）。

    用表表示时必然出现一条接缝，接缝两侧动作相差 80°。
    接缝故意放在 y = 0.037 而不是 y = 0：
    y = 0 正好落在压缩块的边界上（格坐标 15.5，块下标 15//4=3 与 16//4=4），
    跨接缝合并这件事根本不会发生 —— 第一版就是这个原因，
    两种门槛给出完全相同的数字。真实系统里接缝不会对齐块边界，所以必须错开测。
    """
    d = np.atleast_2d(np.asarray(d, dtype=np.float64))
    base = field(d)
    side = np.where(d[:, 1:2] >= 0.037, 1.0, -1.0)
    ang = np.radians(40.0) * side
    c, s = np.cos(ang), np.sin(ang)
    return np.stack([base[:, 0] * c[:, 0] - base[:, 1] * s[:, 0],
                     base[:, 0] * s[:, 0] + base[:, 1] * c[:, 0]],
                    axis=-1).astype(np.float32)


def ang_err(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """两个方向之间的夹角（度）。预测为零向量时记 180°。"""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    pn = np.linalg.norm(p, axis=-1)
    good = pn > 1e-8
    out = np.full(len(p), 180.0)
    if good.any():
        u = p[good] / pn[good][..., None]
        v = t[good] / np.maximum(np.linalg.norm(t[good], axis=-1), 1e-8)[..., None]
        out[good] = np.degrees(np.arccos(np.clip((u * v).sum(-1), -1, 1)))
    return out


def trajectory(rng: np.random.Generator, fn=field) -> list[np.ndarray]:
    """一次真实到达轨迹上的所有 delta。练习和评估都用这个分布。"""
    th = rng.uniform(0, 2 * np.pi)
    r0 = rng.uniform(0.75, 0.95)
    p = np.array([r0 * np.cos(th), r0 * np.sin(th)])
    goal = rng.uniform(-0.05, 0.05, 2)
    out: list[np.ndarray] = []
    for _ in range(120):
        d = goal - p
        if np.linalg.norm(d) <= TOL:
            break
        out.append(d.copy())
        p = p + STEP * fn(d)[0]
    return out


def build_verbatim(episodes: int = N_PRACTICE, fn=field) -> Hierarchy:
    """跑练习，让逐字表长出来。

    一次练习 = 一次真实的到达轨迹。记的是真值动作，不是网络近似，
    这样压缩损失的测量才干净。
    """
    h = Hierarchy(Grid(GRID, EXTENT), 1, 2, min_episodes=3)
    rng = np.random.default_rng(1)
    for e in range(episodes):
        for d in trajectory(rng, fn):
            h.practice(d, fn(d)[0], 0, e, now=e)
    h.consolidate()
    return h


def eval_set(n: int = 120, fn=field) -> tuple[np.ndarray, np.ndarray]:
    """留出评估集：另外跑一批轨迹，取所有 delta。"""
    rng = np.random.default_rng(9999)
    ds = [d for _ in range(n) for d in trajectory(rng, fn)]
    arr = np.stack(ds)
    return arr, fn(arr)


def clone(mem: LUTMemory) -> LUTMemory:
    m = LUTMemory(mem.grid, mem.n_skills, mem.act_dim)
    m.actions = mem.actions.copy()
    m.occupancy = mem.occupancy.copy()
    return m


def build_compressed(block: int = 4, cmin: float = 0.6, fn=field
                     ) -> tuple[Hierarchy, dict]:
    """练习 → 巩固 → 压缩，一步到位。"""
    h = build_verbatim(fn=fn)
    stats = h.compress_now(block=block, min_support=4, coherence_min=cmin)
    return h, stats


def gist_predict(nodes: list, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """要旨预测 + 覆盖率掩码。"""
    pred = np.zeros((len(d), 2), dtype=np.float32)
    hit = np.zeros(len(d), dtype=bool)
    for i, p in enumerate(d):
        a, _ = query_gists(nodes, p, 0)
        if a is not None:
            pred[i] = a
            hit[i] = True
    return pred, hit


def eval_hier(mem: LUTMemory, nodes: list, d: np.ndarray, truth: np.ndarray,
              const: bool = False) -> np.ndarray:
    """在「压缩后的完整系统」上评估：被压的走要旨，没压的仍走逐字。

    const=True 时把要旨降级成块均值（丢掉雅可比），用来量化「主旨保留到一阶」
    到底值多少 —— 这是用户原话「涵盖大体主旨」的最朴素实现。
    """
    h = Hierarchy(Grid(GRID, EXTENT), 1, 2, min_episodes=3)
    h.verbatim = mem
    h.gists = list(nodes)
    pred = np.zeros((len(d), 2), dtype=np.float32)
    for i, p in enumerate(d):
        if const:
            cands = [n for n in nodes if n.covers(p)[0]]
            if cands:
                n0 = min(cands, key=lambda n: float(np.linalg.norm(p - n.center)))
                pred[i] = n0.action0
                continue
        a, _, _ = h.query(p, 0)
        pred[i] = a
    return ang_err(pred, truth)


# ------------------------------------------------------------------ H1 压缩损失


def h1_curvature(d: np.ndarray, truth: np.ndarray) -> dict:
    """压缩损失 = 曲率。对照「仿射要旨」与「均值要旨」。

    评估集固定为**逐字表完全覆盖**的状态，所有块大小同一批状态，
    否则块越大覆盖越多、误差越高只是子集变了，不是压缩变差。
    """
    base = build_verbatim()
    n_cells = int((base.verbatim.occupancy > 0.5).sum())
    per_cell = 4
    _, w0 = base.verbatim.query(d, 0)
    mask = w0 >= 0.999
    dm, tm = d[mask], truth[mask]

    e0 = ang_err(base.verbatim.query(dm, 0)[0], tm)
    rows = [{"block": 0, "radius": 0.0, "nodes": 0, "coverage": 1.0,
             "footprint": 1.0, "affine_err": float(e0.mean()),
             "affine_p95": float(np.percentile(e0, 95)),
             "const_err": float(e0.mean()),
             "curvature": 0.0}]

    for block in (2, 3, 4, 6, 8):
        mem = clone(base.verbatim)
        nodes, _ = compress(mem, block=block, min_support=4, coherence_min=0.6)
        if not nodes:
            continue
        e_aff = eval_hier(mem, nodes, dm, tm, const=False)
        e_const = eval_hier(mem, nodes, dm, tm, const=True)
        fp = int((mem.occupancy > 0.5).sum()) * per_cell + sum(n.n_params for n in nodes)
        cov = float(np.mean(np.any([n.covers(dm, 0.95)[0] for n in nodes], axis=0)))
        rows.append({
            "block": block,
            "radius": float(np.mean([n.radius for n in nodes])),
            "nodes": len(nodes),
            "coverage": cov,
            "footprint": fp / (n_cells * per_cell),
            "affine_err": float(e_aff.mean()),
            "affine_p95": float(np.percentile(e_aff, 95)),
            "const_err": float(e_const.mean()),
            "curvature": float(np.mean([n.curvature for n in nodes])),
        })
    return {"n_cells": n_cells, "n_eval": int(mask.sum()), "rows": rows}


# ------------------------------------------------------------------ H2 可压性


def h2_separation(name: str, fn=field) -> dict:
    """coherence 门槛在不同动作场上的作用。

        平滑场   块内动作缓慢转向，仿射模型吃得下，门槛几乎不触发。
        接缝场   两侧各一条合法通路，跨接缝合并会把两条通路平均成谁都不通。
    """
    base = build_verbatim(fn=fn)
    d, truth = eval_set(80, fn)
    out: dict = {"field": name}

    # 基线：完全不压缩。用来区分「压缩造成的损失」和「表示本身的上限」
    pb, wb = base.verbatim.query(d, 0)
    mb = wb > 0.999
    eb = ang_err(pb[mb], truth[mb])
    out["逐字不压"] = {
        "nodes": 0, "rejected_blocks": 0, "cells_removed": 0,
        "coverage": float(mb.mean()), "mean_err": float(eb.mean()),
        "p95": float(np.percentile(eb, 95)), "max": float(eb.max()),
        "over_90": float((eb > 90).mean()), "over_45": float((eb > 45).mean()),
        "min_coherence": 1.0,
    }

    for label, cmin in (("无门槛", 0.0), ("有门槛", 0.6)):
        mem = clone(base.verbatim)
        nodes, stats = compress(mem, block=4, min_support=4, coherence_min=cmin)

        # 经分级记忆整体查询：被拒绝的块仍留在逐字表里，不能按「只有要旨」算覆盖率
        h = Hierarchy(Grid(GRID, EXTENT), 1, 2, min_episodes=3)
        h.verbatim = mem
        h.gists = list(nodes)
        pred = np.zeros((len(d), 2), dtype=np.float32)
        hit = np.zeros(len(d), dtype=bool)
        for i, p in enumerate(d):
            a, w, _ = h.query(p, 0)
            pred[i] = a
            hit[i] = w > 0
        err = ang_err(pred[hit], truth[hit])
        out[label] = {
            "nodes": len(nodes),
            "rejected_blocks": stats["rejected"],
            "cells_removed": stats["cells_removed"],
            "coverage": float(hit.mean()),
            "mean_err": float(err.mean()),
            "p95": float(np.percentile(err, 95)),
            "max": float(err.max()),
            "over_90": float((err > 90).mean()),
            "over_45": float((err > 45).mean()),
            "min_coherence": float(min((n.coherence for n in nodes), default=0.0)),
        }
    return out


# ------------------------------------------------------------------ H3 再巩固


def h3_reconsolidation(episodes: int = 60, per_ep: int = 6) -> dict:
    """再巩固回填：压缩后的区域重新被用到时，细节怎么长回来。

    三组，门槛固定 min_episodes=3：

        gist_prior   有要旨先验（仿射），没有逐字
        cold         什么都没有（等价于只能靠网络兜底；这里用零向量代表）
        gist_schema  有要旨先验，且被要旨覆盖的格门槛降到 1
                     （Tse et al. 2007 图式同化）。这组是**建模假设**，
                     不是实测事实，单独标出来。

    必须说明：cold 的起点 180° 不是「方法差」，而是这里没有网络兜底，
    零向量被记成 180°。有意义的对比是整条曲线的相对位置。
    """
    base, _ = build_compressed(block=8)
    cands = [n for n in base.gists if float(np.linalg.norm(n.center)) > 0.3]
    if not cands:
        return {"error": "没有合适的要旨节点"}
    node = max(cands, key=lambda n: n.support)
    center = node.center

    ev = []
    rng_ev = np.random.default_rng(4242)
    while len(ev) < 240:
        c = center + rng_ev.normal(0, node.radius * 0.5, 2)
        if np.linalg.norm(c - center) <= node.radius:
            ev.append(c)
    ev = np.array(ev)
    truth_ev = field(ev)

    def err_of(h) -> float:
        pred = np.zeros_like(truth_ev)
        for i, p in enumerate(ev):
            a, _, _ = h.query(p, 0)
            pred[i] = a
        return float(ang_err(pred, truth_ev).mean())

    curves: dict[str, list[float]] = {}
    for label in ("gist_prior", "gist_schema", "cold"):
        h = Hierarchy(Grid(GRID, EXTENT), 1, 2, min_episodes=3)
        if label != "cold":
            h.gists = list(base.gists)
        curve = [err_of(h)]
        rng_ep = np.random.default_rng(777)
        for ep in range(1, episodes + 1):
            pts = ev[rng_ep.integers(0, len(ev), per_ep)]
            for p in pts:
                schema = (label == "gist_schema"
                          and query_gists(h.gists, p, 0)[0] is not None)
                h.min_episodes = 1 if schema else 3
                h.expand(p, 0, field(p)[0], ep)
            curve.append(err_of(h))
        curves[label] = curve

    def first_below(curve, thr=5.0):
        for i, v in enumerate(curve):
            if v < thr:
                return i
        return None

    return {
        "region_radius": float(node.radius),
        "center": [float(v) for v in center],
        "curves": curves,
        "episodes_to_5deg": {k: first_below(v) for k, v in curves.items()},
    }


# ------------------------------------------------------------------ H4 体积-精度


def h4_frontier(d: np.ndarray, truth: np.ndarray) -> dict:
    """体积-精度前沿：块均值 / 仿射 / 逐字。

    只比较三种**无空洞**表示，不做「随机删格」那种稻草人对照：
    删格会留下空洞，插值直接失效，赢它没有意义。

    块均值 = 用户原话「涵盖大体主旨」最朴素的实现；
    仿射 = 额外保留斜率。两者差距就是「主旨该保留到几阶」。
    """
    base = build_verbatim()
    n_cells = int((base.verbatim.occupancy > 0.5).sum())
    per_cell = 4
    _, w0 = base.verbatim.query(d, 0)
    mask = w0 >= 0.999
    dm, tm = d[mask], truth[mask]

    e0 = ang_err(base.verbatim.query(dm, 0)[0], tm)
    rows = [{"mode": "逐字全表", "footprint": 1.0, "err": float(e0.mean())}]

    for block in (2, 3, 4, 6, 8):
        mem = clone(base.verbatim)
        nodes, _ = compress(mem, block=block, min_support=4, coherence_min=0.6)
        if not nodes:
            continue
        fp = (int((mem.occupancy > 0.5).sum()) * per_cell
              + sum(n.n_params for n in nodes))
        rows.append({
            "mode": f"块均值 b={block}", "block": block, "kind": "const",
            "footprint": fp / (n_cells * per_cell),
            "err": float(eval_hier(mem, nodes, dm, tm, const=True).mean()),
        })
        rows.append({
            "mode": f"仿射要旨 b={block}", "block": block, "kind": "affine",
            "footprint": fp / (n_cells * per_cell),
            "err": float(eval_hier(mem, nodes, dm, tm, const=False).mean()),
        })
    return {"n_cells": n_cells, "n_eval": int(mask.sum()), "rows": rows}


def main() -> None:
    out_dir = os.path.dirname(os.path.abspath(__file__))
    d, truth = eval_set()
    print(f"评估集：{len(d)} 个状态（来自 120 条留出轨迹）")

    print("=" * 80)
    print("H1  压缩损失 = 曲率")
    print("=" * 80)
    r1 = h1_curvature(d, truth)
    print(f"逐字表 {r1['n_cells']} 格，评估 {r1['n_eval']} 个被完全覆盖的状态")
    print(f"{'块边长':>6}{'半径':>9}{'节点':>6}{'体积':>8}"
          f"{'仿射':>9}{'p95':>9}{'均值':>9}{'倍数':>7}{'曲率RMS':>10}")
    for r in r1["rows"]:
        print(f"{r['block']:>6}{r['radius']:>9.4f}{r['nodes']:>6}"
              f"{r['footprint']:>8.0%}"
              f"{r['affine_err']:>8.2f}°{r['affine_p95']:>8.2f}°"
              f"{r['const_err']:>8.2f}°"
              f"{r['const_err'] / max(r['affine_err'], 1e-6):>6.1f}x"
              f"{r['curvature']:>10.4f}")

    print()
    print("=" * 80)
    print("H2  coherence 门槛（pattern separation）")
    print("=" * 80)
    r2 = {}
    for name, fn in (("平滑场", field), ("接缝场", field_seam)):
        res = h2_separation(name, fn)
        r2[name] = res
        print(f"\n[{name}]")
        print(f"{' 策略':>8}{'节点':>6}{'拒绝块':>8}{'删格':>6}{'覆盖率':>8}"
              f"{'平均':>9}{'p95':>9}{'最大':>9}{'>90°':>8}{'>45°':>8}")
        for k in ("逐字不压", "无门槛", "有门槛"):
            v = res[k]
            print(f"{k:>8}{v['nodes']:>6}{v['rejected_blocks']:>8}"
                  f"{v['cells_removed']:>6}{v['coverage']:>8.1%}"
                  f"{v['mean_err']:>8.2f}°{v['p95']:>8.2f}°{v['max']:>8.2f}°"
                  f"{v['over_90']:>7.1%}{v['over_45']:>7.1%}")

    print()
    r3 = h3_reconsolidation()
    print("=" * 80)
    print("H3  再巩固回填（区域半径 %.3f，每次再遇到 6 个样本）" % r3["region_radius"])
    print("=" * 80)
    print(f"{'再遇到':>8}{'要旨先验':>12}{'要旨+图式':>12}{'冷启动':>12}")
    for i in range(0, len(r3["curves"]["gist_prior"]), 5):
        print(f"{i:>8}{r3['curves']['gist_prior'][i]:>11.2f}°"
              f"{r3['curves']['gist_schema'][i]:>11.2f}°"
              f"{r3['curves']['cold'][i]:>11.2f}°")
    print(f"  首次低于 5°：{r3['episodes_to_5deg']}")
    c3 = r3["curves"]
    t3 = {k: next((i for i, v in enumerate(c) if v < 3.0), None) for k, c in c3.items()}
    print(f"  首次低于 3°：{t3}")

    print()
    print("=" * 80)
    print("H4  体积-精度前沿（三种都无空洞）")
    print("=" * 80)
    r4 = h4_frontier(d, truth)
    print(f"{'表示':>16}{'体积':>8}{'误差':>9}")
    for r in r4["rows"]:
        print(f"{r['mode']:>16}{r['footprint']:>8.0%}{r['err']:>8.2f}°")

    data = {"h1": r1, "h2": r2, "h3": r3, "h4": r4}
    path = os.path.join(out_dir, "hierarchy.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n数据已写入 {path}")


if __name__ == "__main__":
    main()
