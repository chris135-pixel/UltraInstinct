"""Ultra Instinct —— PyTorch 在控制环中的真实单步延迟。

控制环与离线推理的区别：每一步状态都是新数据，必须写进张量。因此
「新建 tensor → forward → 取标量」这条完整链路才是真实成本，
只测 model(x) 会低估。

同时测三种写法：
    raw     每次 torch.tensor(state) 新建输入（最朴素的写法）
    buf     预分配 buffer + copy_ 写入（工程优化写法）
    np2tor  numpy 状态先转张量再前向（传感器给 numpy 时的真实情况）
"""

from __future__ import annotations

import json
import time
import statistics

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)  # 单步推理不需要多线程，线程池调度反而是开销

RNG = np.random.default_rng(0)


class TinyMLP(nn.Module):
    """2 → 64 → 64 → 2，与原方案规模一致。"""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


model = TinyMLP().eval()
state_np = RNG.normal(0, 1, 2).astype(np.float32)
state_t = torch.from_numpy(state_np.copy())
buf = torch.empty(1, 2)
grad_buf = torch.empty(1, 2, requires_grad=True)


def bench(fn, iters: int = 20_000, warmup: int = 3_000) -> dict[str, float]:
    """稳态单步延迟，单位 ns。"""
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


@torch.no_grad()
def fwd_raw() -> float:
    """每次新建 tensor（朴素写法）。"""
    x = torch.tensor([[0.37, -0.82]])
    return float(model(x)[0, 0])


@torch.no_grad()
def fwd_buf() -> None:
    """预分配 buffer + copy_，避免分配开销。"""
    buf.copy_(state_t)
    model(buf)


@torch.no_grad()
def fwd_np2tor() -> float:
    """numpy 状态转张量后前向（传感器输出 numpy 的真实情况）。"""
    x = torch.from_numpy(state_np)
    return float(model(x.unsqueeze(0))[0, 0])


def fwd_no_nograd() -> None:
    """忘记 no_grad 的代价：autograd 会建计算图。"""
    model(buf)


def fwd_requires_grad() -> None:
    """显式 requires_grad 的张量：建图 + 记录。"""
    model(grad_buf)


@torch.no_grad()
def batch_forward(batch: int) -> float:
    """批量前向的每样本延迟（ns）。"""
    x = torch.randn(batch, 2)
    iters = max(200, 20_000 // batch)
    for _ in range(max(50, iters // 10)):
        model(x)
    samples: list[int] = []
    chunk = max(1, min(200, iters))
    for _ in range(iters // chunk):
        t0 = time.perf_counter_ns()
        for _ in range(chunk):
            model(x)
        samples.append((time.perf_counter_ns() - t0) // chunk)
    return statistics.median(samples) / batch


def main() -> None:
    print(f"torch {torch.__version__} | threads={torch.get_num_threads()} | "
          f"interop={torch.get_num_interop_threads()}")
    print(f"CPU: {torch.backends.cpu.get_cpu_capability()}")
    print("=" * 70)
    rows = {}
    for name, fn in (
        ("torch · 新建 tensor + forward", fwd_raw),
        ("torch · 预分配 buffer + forward", fwd_buf),
        ("torch · numpy→tensor + forward", fwd_np2tor),
        ("torch · 漏用 no_grad（建图）", fwd_no_nograd),
        ("torch · 输入 requires_grad", fwd_requires_grad),
    ):
        r = bench(fn)
        rows[name] = r["p50"]
        print(f"{name:<34}{r['p50']:>9.0f} ns  (mean {r['mean']:.0f}, p99 {r['p99']:.0f})")

    print("-" * 70)
    batch_rows = {}
    for b in (1, 8, 64, 512, 2048):
        ns = batch_forward(b)
        batch_rows[b] = ns
        print(f"torch · batch={b:<6} 每样本 {ns:>8.1f} ns")

    with open("benchmarks/torch_latency.json", "w", encoding="utf-8") as f:
        json.dump({"single": rows, "batch_per_sample": batch_rows}, f,
                  ensure_ascii=False, indent=2)
    print("\n已写入 benchmarks/torch_latency.json")


if __name__ == "__main__":
    main()
