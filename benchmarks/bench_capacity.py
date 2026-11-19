"""Ultra Instinct —— 能力-速度前沿：模型层到底能做到多快。

「能力」在这个任务里拆成三个可测维度：
    1. 输入维度   —— 状态表达力（2 维 Δ → 512 维视觉特征）
    2. 网络宽度   —— 函数逼近能力
    3. 泛化要求   —— 能否插值到没见过的状态（决定 LUT 是否可用）

本脚本测的是「单步前向」的纯模型延迟，不含任何 I/O。
"""

from __future__ import annotations

import json
import statistics
import time
from array import array

import numpy as np

RNG = np.random.default_rng(11)


def bench(fn, iters: int = 20_000, warmup: int = 3_000) -> dict[str, float]:
    """稳态单步延迟（ns）。"""
    for _ in range(warmup):
        fn()
    samples: list[int] = []
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
    }


# ---------------------------------------------------------------- numpy MLP 规模扫描

SPECS = (
    ("极小 2-16-16-2", (2, 16, 16, 2)),
    ("小   2-64-64-2", (2, 64, 64, 2)),
    ("中   8-128-128-4", (8, 128, 128, 4)),
    ("大   64-256-256-7", (64, 256, 256, 7)),
    ("视觉 512-512-512-12", (512, 512, 512, 12)),
)


def build_numpy(d_in: int, h1: int, h2: int, d_out: int):
    """构造 numpy 权重与输入。"""
    ws = (
        RNG.normal(0, 0.4, (d_in, h1)).astype(np.float32), np.zeros(h1, np.float32),
        RNG.normal(0, 0.25, (h1, h2)).astype(np.float32), np.zeros(h2, np.float32),
        RNG.normal(0, 0.25, (h2, d_out)).astype(np.float32), np.zeros(d_out, np.float32),
    )
    x = RNG.normal(0, 1, d_in).astype(np.float32)
    return ws, x


def numpy_fwd(ws, x):
    """numpy 前向，输入保持 1 维以避免 batch 维开销。"""
    W1, B1, W2, B2, W3, B3 = ws
    h1 = np.tanh(x @ W1 + B1)
    h2 = np.tanh(h1 @ W2 + B2)
    return h2 @ W3 + B3


# ---------------------------------------------------------------- torch 对照

def torch_fwd_factory(d_in: int, h1: int, h2: int, d_out: int):
    """构造 torch 版本，返回无参闭包（预分配 buffer）。"""
    import torch
    import torch.nn as nn

    torch.set_num_threads(1)
    net = nn.Sequential(
        nn.Linear(d_in, h1), nn.Tanh(),
        nn.Linear(h1, h2), nn.Tanh(),
        nn.Linear(h2, d_out),
    ).eval()
    buf = torch.empty(1, d_in)

    @torch.no_grad()
    def run():
        net(buf)

    return run


# ---------------------------------------------------------------- LUT 维度扫描

def lut_2d() -> float:
    """128x128 二维表，32KB，常驻 L1。"""
    return _t2[_IX][_IY]


def lut_3d() -> float:
    """32x32x32 三维表，64KB，贴 L1/L2 边界。"""
    return _t3[_IX3][_IY3][_IZ3]


def lut_4d() -> float:
    """16^4 四维表，128KB，进 L2。"""
    return _t4[_I4][_J4][_K4][_L4]


G2, G3, G4 = 128, 32, 16
_t2 = [[RNG.normal() for _ in range(G2)] for _ in range(G2)]
_t3 = [[[RNG.normal() for _ in range(G3)] for _ in range(G3)] for _ in range(G3)]
_t4 = [[[[RNG.normal() for _ in range(G4)] for _ in range(G4)]
        for _ in range(G4)] for _ in range(G4)]
_IX, _IY = 91, 37
_IX3, _IY3, _IZ3 = 20, 11, 27
_I4, _J4, _K4, _L4 = 9, 3, 12, 7

# array('f') 扁平布局，模拟 C 数组
_flat2 = array("f", RNG.normal(0, 1, G2 * G2).astype(np.float32))


def lut_flat_2d() -> float:
    """扁平一维定址：k = i*G + j，最接近编译产物。"""
    return _flat2[_IX * G2 + _IY]


# ---------------------------------------------------------------- 主流程


def main() -> None:
    print("=" * 78)
    print("[A] 纯模型单步延迟 vs 网络规模（numpy float32 / torch eager CPU）")
    print("=" * 78)
    print(f"{'规模':<22}{'numpy':>12}{'torch':>12}{'倍数':>8}{'MAC 数':>12}")
    rows = []
    for name, spec in SPECS:
        ws, x = build_numpy(*spec)
        r_np = bench(lambda: numpy_fwd(ws, x), iters=10_000 if spec[0] < 256 else 2_000)
        try:
            t_fn = torch_fwd_factory(*spec)
            r_t = bench(t_fn, iters=5_000 if spec[0] < 256 else 1_000)
            t_ns = r_t["p50"]
        except Exception as exc:  # noqa: BLE001
            t_ns = float("nan")
            print(f"  torch 跳过: {exc}")
        mac = spec[0] * spec[1] + spec[1] * spec[2] + spec[2] * spec[3]
        rows.append({"spec": name, "numpy_ns": r_np["p50"], "torch_ns": t_ns, "mac": mac})
        print(f"{name:<22}{r_np['p50']:>10.0f}n{t_ns:>10.0f}n{t_ns / r_np['p50']:>7.1f}x{mac:>12,}")

    print()
    print("=" * 78)
    print("[B] 记忆表维度扫描（O(1) 索引，差别只在 cache 层次）")
    print("=" * 78)
    lut_rows = {}
    for name, fn, note in (
        ("2D 128x128 (32KB, L1)", lut_2d, "16,384 格"),
        ("3D 32^3 (64KB, L1/L2)", lut_3d, "32,768 格"),
        ("4D 16^4 (128KB, L2)", lut_4d, "65,536 格"),
        ("扁平 array 2D (32KB)", lut_flat_2d, "C 数组布局"),
    ):
        r = bench(fn)
        lut_rows[name] = r["p50"]
        print(f"{name:<26}{r['p50']:>8.0f} ns   p99 {r['p99']:>5.0f}   {note}")

    print()
    print("=" * 78)
    print("[C] 混合通路的平均延迟：avg = p*LUT + (1-p)*MLP")
    print("=" * 78)
    lut_ns = lut_rows["2D 128x128 (32KB, L1)"]
    mlp_ns = next(r["numpy_ns"] for r in rows if "2-64-64-2" in r["spec"])
    print(f"LUT={lut_ns:.0f}ns  MLP={mlp_ns:.0f}ns")
    print(f"{'命中率 p':>10}{'平均延迟':>14}{'有效频率':>14}")
    mix_rows = []
    for p in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0):
        avg = p * lut_ns + (1 - p) * mlp_ns
        mix_rows.append({"hit_rate": p, "avg_ns": avg})
        print(f"{p:>10.2f}{avg:>11.0f} ns{1e9 / avg / 1000:>11.0f} kHz")

    payload = {"scale": rows, "lut": lut_rows, "mix": mix_rows}
    with open("benchmarks/capacity_latency.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("\n已写入 benchmarks/capacity_latency.json")


if __name__ == "__main__":
    main()
