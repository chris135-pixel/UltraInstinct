"""端到端演示：二维到达任务上完整的「练习 → 固化 → 使用」流程。

跑法：
    python examples/reach_demo.py

输出四组对比：
    纯网络      没有记忆，每次都现算
    纯记忆      只查表，查不到就摆烂（用来暴露「表不泛化」的代价）
    混合        查表为主，边界带和未覆盖区走网络
    混合+自检   再加「连续两次没进展就退回纯网络」
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultra_instinct.consolidation import CandidatePool, compile_memory
from ultra_instinct.memory import LUTMemory
from ultra_instinct.policy import Policy, PolicyConfig
from ultra_instinct.quantize import Grid
from ultra_instinct.training import ReplayBuffer, TorchMLP, TrainConfig, export_mlp, train_skill

N_SKILLS = 1
GRID = 16
EXTENT = 1.0
STEP = 0.03
TOL = 0.01
MAX_STEPS = 150
PRACTICE_EPISODES = 300
EVAL_EPISODES = 120
MIN_EPISODES = 5


def field(d: np.ndarray) -> np.ndarray:
    """带曲率的动作场：中远距离绕行，近目标收敛直行。

    曲率在近目标处衰减，否则任何实现都会绕圈 —— 那是任务设计问题。
    """
    r = np.linalg.norm(d, axis=-1)
    ang = 1.05 * np.exp(-r / 0.35) / (1.0 + np.exp(-(r - 0.35) / 0.07))
    c, s = np.cos(ang), np.sin(ang)
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    u = d / np.maximum(n, 1e-8)
    return np.stack([u[..., 0] * c - u[..., 1] * s,
                     u[..., 0] * s + u[..., 1] * c], axis=-1).astype(np.float32)


def sample_start(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    th = rng.uniform(0, 2 * np.pi, n)
    r = rng.uniform(0.6, 0.95, n)
    pos = np.stack([r * np.cos(th), r * np.sin(th)], axis=-1)
    goal = rng.uniform(-0.05, 0.05, (n, 2))
    return pos, goal


def run_episode(policy, pos, goal, selfcheck: bool = False,
                record: CandidatePool | None = None, episode_id: int = 0,
                skill_id: int = 0) -> dict:
    """跑一个到达 episode。

    返回是否到达、最终误差、路径长度、网络调用次数。
    """
    policy.begin_episode(skill_id)
    p = pos.copy()
    dist = 0.0
    calls = 0
    steps = 0
    for _ in range(MAX_STEPS):
        delta = goal - p
        remaining = float(np.linalg.norm(delta))
        if remaining <= TOL:
            break
        out = policy.act(delta, skill_id)
        calls += int(out.net_called)
        steps += 1

        step = min(STEP, remaining)
        nxt = p + step * out.action
        dist += float(np.linalg.norm(nxt - p))
        if record is not None:
            record.record(delta, skill_id, out.action, episode_id)
        p = nxt

    final = float(np.linalg.norm(goal - p))
    reached = final <= TOL
    if selfcheck:
        policy.end_episode(reached, skill_id)
    return {
        "reached": reached,
        "final_err": final,
        "path": dist,
        "straight": float(np.linalg.norm(pos - goal)),
        "net_calls": calls,
        "steps": max(steps, 1),
    }


def evaluate(policy, label: str, rng, selfcheck: bool = False) -> dict:
    """批量评估，并计单步耗时。"""
    pos, goal = sample_start(rng, EVAL_EPISODES)
    policy.reset()
    inval_base = sum(policy.invalidated_count.values())

    rows = []
    t_total = 0.0
    for i in range(EVAL_EPISODES):
        t0 = time.perf_counter()
        r = run_episode(policy, pos[i], goal[i], selfcheck=selfcheck)
        t_total += time.perf_counter() - t0
        rows.append(r)

    reached = np.array([r["reached"] for r in rows])
    err = np.array([r["final_err"] for r in rows]) * 1000
    calls = np.array([r["net_calls"] for r in rows], dtype=float)
    steps = np.array([r["steps"] for r in rows], dtype=float)
    path_ratio = np.array([r["path"] / r["straight"] for r in rows])
    return {
        "label": label,
        "reached": float(reached.mean()),
        "err": float(err.mean()),
        "err_p95": float(np.percentile(err, 95)),
        "net_call_rate": float(calls.sum() / steps.sum()),
        "calls_per_ep": float(calls.mean()),
        "path_ratio": float(path_ratio.mean()),
        "us_per_ep": t_total / EVAL_EPISODES * 1e6,
        "invalidated": sum(policy.invalidated_count.values()) - inval_base,
    }


def main() -> None:
    rng = np.random.default_rng(0)
    hidden = 64
    input_dim = 2 + 1

    print("=" * 74)
    print("Ultra Instinct — 二维到达任务端到端演示")
    print("=" * 74)

    # ------------------------------------------------ 1. 练习：训练慢通路
    print("\n[1/4] 训练慢通路网络（torch，仅离线）")
    torch_net = TorchMLP(input_dim=input_dim, hidden=hidden, output_dim=2)
    replay = ReplayBuffer()
    cfg = TrainConfig(hidden=hidden, steps=1500, lr=2e-3, batch=256)
    loss = train_skill(torch_net, field, cfg, rng, skill_id=0,
                       n_skills=N_SKILLS, replay=replay)
    net = export_mlp(torch_net)
    print(f"      最终损失 {loss:.5f}，参数量 "
          f"{sum(p.numel() for p in torch_net.parameters())}")

    cfg_policy = PolicyConfig(n_skills=N_SKILLS, grid_size=GRID, extent=EXTENT,
                              max_turn_deg=8.0)
    pure_net = Policy(net=net, memory=None, config=cfg_policy)

    # ------------------------------------------------ 2. 练习 episode → 候选池
    print(f"\n[2/4] 跑 {PRACTICE_EPISODES} 个练习 episode，收集成功经验")
    pool = CandidatePool(Grid(GRID, EXTENT), n_skills=N_SKILLS,
                         min_episodes=MIN_EPISODES)
    pos, goal = sample_start(rng, PRACTICE_EPISODES)
    n_ok = 0
    for i in range(PRACTICE_EPISODES):
        pure_net.reset()
        r = run_episode(pure_net, pos[i], goal[i], record=pool, episode_id=i)
        n_ok += int(r["reached"])
    st = pool.stats()
    print(f"      成功 {n_ok}/{PRACTICE_EPISODES}，触及 {st['cells_touched']} 格，"
          f"够条件 {st['cells_ready']} 格（占 {st['ready_ratio']:.1%}）")

    # ------------------------------------------------ 3. 固化
    print(f"\n[3/4] 固化：跨 {MIN_EPISODES} 个 episode 成功且动作一致的格子入表")
    entries = pool.promoted()
    memory = compile_memory(entries, Grid(GRID, EXTENT), N_SKILLS, act_dim=2)
    print(f"      条目 {len(entries)} 条，覆盖率 {memory.coverage[0]:.1%}")

    # ------------------------------------------------ 4. 对比
    ckpt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "checkpoints", "reach")
    mixed = Policy(net=net, memory=memory, config=cfg_policy)
    mixed.save(ckpt)

    print(f"\n[4/4] 评估 {EVAL_EPISODES} 个 episode（已保存到 {ckpt}）")
    lut_only = Policy(net=None, memory=memory, config=cfg_policy)

    results = [
        evaluate(pure_net, "纯网络", np.random.default_rng(7)),
        evaluate(lut_only, "纯记忆（无兜底）", np.random.default_rng(7)),
        evaluate(mixed, "混合", np.random.default_rng(7)),
        evaluate(mixed, "混合 + 自检", np.random.default_rng(7), selfcheck=True),
    ]

    print(f"\n{'方案':<20}{'到达率':>8}{'误差':>9}{'p95':>9}{'网络调用':>10}{'路径/直线':>11}{'撤销':>7}")
    print("-" * 74)
    for r in results:
        print(f"{r['label']:<20}{r['reached']:>8.1%}{r['err']:>8.1f}mm"
              f"{r['err_p95']:>8.1f}mm{r['net_call_rate']:>10.1%}"
              f"{r['path_ratio']:>11.3f}{r['invalidated']:>7}")

    base = results[0]
    mix = results[2]
    print(f"\n混合相对纯网络：网络调用降 "
          f"{(1 - mix['net_call_rate'] / max(base['net_call_rate'], 1e-6)):.0%}，"
          f"精度变化 {mix['err'] - base['err']:+.1f}mm")


if __name__ == "__main__":
    main()
