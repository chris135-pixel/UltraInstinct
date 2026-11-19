"""Ultra Instinct —— 优化空间实测：批量吞吐、静态表编译、容量-延迟曲线。

回答三个问题：
    1. 记忆库变大时延迟如何变化（用户想看的「响应时间随固化下降」是否会反向）。
    2. 批量推理下，网络 vs 查表谁更快（多关节 / 高控制频率场景）。
    3. 把记忆「编译」成静态表后，延迟地板在哪（自在极意的工程上限）。
"""

from __future__ import annotations

import json
import time
from array import array
from typing import Callable

import numpy as np

from bench_latency import bench, make_mlp

RNG = np.random.default_rng(7)
W1, B1, W2, B2, W3, B3 = make_mlp()


# ---------------------------------------------------------------- 1. 容量-延迟曲线


def knn_latency_curve() -> dict[int, float]:
    """暴力 KNN 延迟随记忆库条目数的变化。"""
    q = RNG.normal(0, 1, 2).astype(np.float32)
    out: dict[int, float] = {}
    for n in (100, 300, 1_000, 3_000, 10_000, 30_000, 100_000):
        keys = RNG.normal(0, 1, (n, 2)).astype(np.float32)
        vals = RNG.normal(0, 1, (n, 2)).astype(np.float32)

        def run(keys=keys, vals=vals) -> np.ndarray:
            d = ((keys - q) ** 2).sum(axis=1)
            return vals[int(np.argmin(d))]

        iters = 5_000 if n <= 10_000 else 500
        out[n] = bench(run, iters=iters, warmup=200)["p50"]
    return out


# ---------------------------------------------------------------- 2. 批量吞吐


def batch_throughput(batch: int) -> float:
    """MLP 在给定 batch 下的每样本延迟（ns）。"""
    x = RNG.normal(0, 1, (batch, 2)).astype(np.float32)

    def run() -> np.ndarray:
        h1 = np.tanh(x @ W1 + B1)
        h2 = np.tanh(h1 @ W2 + B2)
        return h2 @ W3 + B3

    iters = max(200, 20_000 // batch)
    total = bench(run, iters=iters, warmup=max(50, iters // 10))["mean"]
    return total / batch


def knn_batch_throughput(batch: int, n: int = 10_000) -> float:
    """KNN 在给定 batch 下的每样本延迟（ns）。需要 (batch, N) 的距离矩阵。"""
    keys = RNG.normal(0, 1, (n, 2)).astype(np.float32)
    vals = RNG.normal(0, 1, (n, 2)).astype(np.float32)
    q = RNG.normal(0, 1, (batch, 2)).astype(np.float32)

    def run() -> np.ndarray:
        d = ((q[:, None, :] - keys[None, :, :]) ** 2).sum(axis=2)  # (batch, N)
        return vals[np.argmin(d, axis=1)]

    iters = max(100, 10_000 // batch)
    total = bench(run, iters=iters, warmup=max(30, iters // 10))["mean"]
    return total / batch


# ---------------------------------------------------------------- 3. 静态表编译（LUT）

GRID = 128
_SCALE = GRID / 2.0

# 定点量化：把 Δ∈[-1,1] 映射到 [0, GRID-1]
_lut_flat = RNG.normal(0, 1, (GRID * GRID * 2)).astype(np.float32)
_lut_list: list[list[tuple[float, float]]] = [
    [(float(RNG.normal()), float(RNG.normal())) for _ in range(GRID)] for _ in range(GRID)
]
_lut_array = array("f", RNG.normal(0, 1, GRID * GRID * 2).astype(np.float32))

_qx, _qy = 0.37, -0.82


def lut_list_lookup() -> tuple[float, float]:
    """嵌套 list 索引：预量化键 → 两次数组索引，无距离计算、无分支。"""
    i = int((_qx + 1.0) * _SCALE)
    j = int((_qy + 1.0) * _SCALE)
    return _lut_list[i][j]


def lut_array_lookup() -> tuple[float, float]:
    """array('f') 一维定址：与 C 数组布局一致，最接近编译产物的形式。"""
    i = int((_qx + 1.0) * _SCALE)
    j = int((_qy + 1.0) * _SCALE)
    k = (i * GRID + j) * 2
    return _lut_array[k], _lut_array[k + 1]


def lut_full_pipeline() -> tuple[float, float]:
    """完整决策链路：量化 + 查表 + 有效性判断 + 未命中回退到网络。"""
    i = int((_qx + 1.0) * _SCALE)
    j = int((_qy + 1.0) * _SCALE)
    hit = _lut_list[i][j]
    if hit is not None:
        return hit
    h1 = np.tanh(np.array([_qx, _qy], np.float32) @ W1 + B1)
    h2 = np.tanh(h1 @ W2 + B2)
    y = h2 @ W3 + B3
    return float(y[0]), float(y[1])


# ---------------------------------------------------------------- 主流程


def main() -> None:
    """输出三组实验数据。"""
    print("=" * 74)
    print("[1] 记忆库容量 → 单步延迟（暴力 KNN，2 维键，float32）")
    print("=" * 74)
    curve = knn_latency_curve()
    print(f"{'条目数':>10}{'单步延迟':>14}{'相对 MLP(9.2us)':>18}")
    for n, ns in curve.items():
        print(f"{n:>10}{ns / 1000:>11.1f} us{ns / 9212:>17.1f}x")

    print()
    print("=" * 74)
    print("[2] 批量吞吐：MLP vs KNN(N=1e4)，单位 ns / 每条")
    print("=" * 74)
    print(f"{'batch':>8}{'MLP':>12}{'KNN':>12}{'倍数':>10}")
    batch_rows = []
    for b in (1, 4, 16, 64, 256, 1024):
        m = batch_throughput(b)
        k = knn_batch_throughput(b, n=10_000)
        batch_rows.append({"batch": b, "mlp_ns": m, "knn_ns": k})
        print(f"{b:>8}{m:>11.1f}{k:>11.1f}{k / m:>9.0f}x")

    print()
    print("=" * 74)
    print("[3] 静态表编译（LUT 128x128，C 数组布局）vs 检索")
    print("=" * 74)
    lut_rows = {}
    for name, fn in (
        ("嵌套 list 索引", lut_list_lookup),
        ("array('f') 定址", lut_array_lookup),
        ("完整链路(量化+查表+分支)", lut_full_pipeline),
    ):
        r = bench(fn)
        lut_rows[name] = r["p50"]
        print(f"{name:<30}{r['p50']:>8.0f} ns   (mean {r['mean']:.0f}, p99 {r['p99']:.0f})")

    payload = {
        "knn_capacity_curve_p50_ns": curve,
        "batch": batch_rows,
        "lut_p50_ns": lut_rows,
    }
    with open("benchmarks/scaling_latency.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("\n已写入 benchmarks/scaling_latency.json")


if __name__ == "__main__":
    main()
