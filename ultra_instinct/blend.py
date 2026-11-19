"""通道路由：把「练熟的动作」和「现算的动作」连续混合。

设计依据（实测，这是整个项目最关键的一条）：

    固化区边界处，两条指令实测夹角
        硬切换           平均 96.2°，最大 103.7°，100% 超过 30°
        连续混合         平均 6.0°，最大 10.3°，0% 超过 30°

而且权重必须来自「固化掩码的插值」，不能用「到最近格中心的距离」。
那个距离本身在格边界上不连续，拿它做权重等于没平滑（实测跳变仍是 90°）。
"""

from __future__ import annotations

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-8)


def blend(lut_action: np.ndarray, net_action: np.ndarray, weight: np.ndarray,
          normalize: bool = True, degenerate_eps: float = 0.2) -> np.ndarray:
    """按固化权重混合两路动作。

    Args:
        lut_action: (n, act_dim) 查表得到的动作。
        net_action: (n, act_dim) 网络现算的动作。
        weight: (n,) 或标量，0 = 完全没练过，1 = 完全练熟。
        normalize: 是否把结果归一化成单位向量（方向型动作用）。
        degenerate_eps: 混合后模长低于此值时不归一化。
            两路夹角接近 180° 时模长趋近 0，此时归一化会把噪声放大成随机方向。

    Returns:
        (n, act_dim) 混合后的动作。
    """
    a = np.atleast_2d(np.asarray(lut_action, dtype=np.float32))
    b = np.atleast_2d(np.asarray(net_action, dtype=np.float32))
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim == 0:
        w = np.full((a.shape[0],), float(w), dtype=np.float32)
    w = w[:, None]

    mixed = w * a + (1.0 - w) * b
    if not normalize:
        return mixed

    norm = np.linalg.norm(mixed, axis=-1, keepdims=True)
    safe = norm > degenerate_eps
    out = np.where(safe, mixed / np.maximum(norm, 1e-8), mixed)
    return out.astype(np.float32)
