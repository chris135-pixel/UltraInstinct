"""Ultra Instinct —— 快通路延迟实测基准。

目的：用真实数据回答一个问题——"查记忆库"是否真的比"小网络前向"更快。

测量方法：
    单步延迟用「批量循环总时间 / 循环次数」计算。不逐次调用 perf_counter_ns，
    因为 Windows QPC 自身约 100ns 开销，逐次计时会严重污染微秒级操作。
    批量平均得到的是稳态单步延迟，正是「每一步控制周期」要付出的代价。

    latency_ns = total_ns / iters      （含 Python 解释器与函数调用开销）
"""

from __future__ import annotations

import json
import statistics
import time
from typing import Callable, Sequence

import numpy as np

# ---------------------------------------------------------------- 计时工具


def bench(fn: Callable[[], object], iters: int = 20_000, warmup: int = 5_000) -> dict[str, float]:
    """测量函数的稳态单步延迟。

    Args:
        fn: 无参可调用对象，内部完成一次完整推理。
        iters: 计时循环次数。
        warmup: 预热次数，用于让 CPU 频率、缓存、分支预测进入稳态。

    Returns:
        含 mean / p50 / p99 / min 的延迟字典，单位纳秒。
    """
    for _ in range(warmup):
        fn()

    samples: list[int] = []
    # 分块计时，避免单次总时长受系统调度抖动影响过大，同时能估计分位数
    chunk = max(1, min(1_000, iters))
    for _ in range(iters // chunk):
        t0 = time.perf_counter_ns()
        for _ in range(chunk):
            fn()
        samples.append((time.perf_counter_ns() - t0) // chunk)

    return {
        "mean": statistics.fmean(samples),
        "p50": statistics.median(samples),
        "p99": sorted(samples)[int(len(samples) * 0.99) - 1],
        "min": min(samples),
    }


# ---------------------------------------------------------------- 慢网络：2 层 MLP

RNG = np.random.default_rng(0)


def make_mlp(d_in: int = 2, h: int = 64, d_out: int = 2) -> tuple[np.ndarray, ...]:
    """构造一个小 MLP 的权重（float32，与实际训练规模一致）。"""
    return (
        RNG.normal(0, 0.5, (d_in, h)).astype(np.float32), np.zeros(h, np.float32),
        RNG.normal(0, 0.3, (h, h)).astype(np.float32), np.zeros(h, np.float32),
        RNG.normal(0, 0.3, (h, d_out)).astype(np.float32), np.zeros(d_out, np.float32),
    )


W1, B1, W2, B2, W3, B3 = make_mlp()
x1d = RNG.normal(0, 1, 2).astype(np.float32)
x2d = x1d.reshape(1, 2)


def mlp_forward_1d() -> np.ndarray:
    """单条状态前向：输入 (2,)，完全展开，无循环。"""
    h1 = np.tanh(x1d @ W1 + B1)
    h2 = np.tanh(h1 @ W2 + B2)
    return h2 @ W3 + B3


def mlp_forward_2d() -> np.ndarray:
    """单条状态前向：输入 (1,2)，保持 batch 维。"""
    h1 = np.tanh(x2d @ W1 + B1)
    h2 = np.tanh(h1 @ W2 + B2)
    return h2 @ W3 + B3


def mlp_forward_np_single_nobatch() -> np.ndarray:
    """用 einsum 手写矩阵乘，绕开 BLAS 对极小矩阵的调度开销。"""
    h1 = np.tanh(np.einsum("i,ij->j", x1d, W1) + B1)
    h2 = np.tanh(np.einsum("i,ij->j", h1, W2) + B2)
    return np.einsum("i,ij->j", h2, W3) + B3


# ---------------------------------------------------------------- 快通路：记忆库

# 网格哈希：把 Δ 量化到 256x256 的格子，O(1) 定位
GRID = 256
grid_table = RNG.normal(0, 1, (GRID, GRID, 2)).astype(np.float32)
q = np.array([0.37, -0.82], dtype=np.float32)


def grid_lookup_numpy() -> np.ndarray:
    """网格哈希查表：两次归一化 + 索引，无距离计算。"""
    idx = ((q + 1.0) * (GRID / 2)).astype(np.int32)
    i, j = int(idx[0]) % GRID, int(idx[1]) % GRID
    return grid_table[i, j]


def dict_lookup() -> np.ndarray:
    """预量化 key 的 dict 查表：纯 O(1) 哈希访问。"""
    return grid_table[91, 23]


# 暴力 KNN：模拟「无索引的记忆库」
def make_knn(n: int) -> tuple[np.ndarray, np.ndarray]:
    keys = RNG.normal(0, 1, (n, 2)).astype(np.float32)
    vals = RNG.normal(0, 1, (n, 2)).astype(np.float32)
    return keys, vals


KNN_SETS = {n: make_knn(n) for n in (1_000, 10_000, 100_000)}


def make_knn_fn(n: int) -> Callable[[], np.ndarray]:
    keys, vals = KNN_SETS[n]

    def run() -> np.ndarray:
        d = ((keys - q) ** 2).sum(axis=1)
        return vals[int(np.argmin(d))]

    return run


# ---------------------------------------------------------------- 纯开销基线


def empty_call() -> None:
    """空函数：量化 Python 层函数调用本身的下限。"""


def baseline_arith() -> float:
    """几十次浮点运算的下限，用于对比。"""
    return 0.37 * 2.0 + 0.5


# ---------------------------------------------------------------- 主流程


def main() -> None:
    """跑完全部基准并打印 Markdown 表格 + JSON。"""
    cases: Sequence[tuple[str, Callable[[], object]]] = (
        ("基线 · 空函数调用", empty_call),
        ("基线 · 2 次浮点运算", baseline_arith),
        ("慢网络 · MLP 2-64-64-2 (1d, matmul)", mlp_forward_1d),
        ("慢网络 · MLP 2-64-64-2 (1x2, matmul)", mlp_forward_2d),
        ("慢网络 · MLP 2-64-64-2 (einsum)", mlp_forward_np_single_nobatch),
        ("快通路 · 网格哈希查表", grid_lookup_numpy),
        ("快通路 · 预量化 dict 查表", dict_lookup),
        ("快通路 · 暴力 KNN N=1e3", make_knn_fn(1_000)),
        ("快通路 · 暴力 KNN N=1e4", make_knn_fn(10_000)),
        ("快通路 · 暴力 KNN N=1e5", make_knn_fn(100_000)),
    )

    results: dict[str, dict[str, float]] = {}
    print(f"{'case':<42}{'mean':>9}{'p50':>9}{'p99':>9}{'min':>9}")
    print("-" * 78)
    for name, fn in cases:
        r = bench(fn)
        results[name] = r
        print(f"{name:<42}{r['mean']:>8.0f}n{r['p50']:>8.0f}n{r['p99']:>8.0f}n{r['min']:>8.0f}n")

    with open("benchmarks/numpy_latency.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n已写入 benchmarks/numpy_latency.json")


if __name__ == "__main__":
    main()
