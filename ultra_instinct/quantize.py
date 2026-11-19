"""网格量化：连续状态 → 整数格坐标 + 插值权重。

设计依据（实测）：

- 用双线性插值，不要取最近格。
  16×16 的表 + 插值实测 1.7mm 误差，优于 128×128 表直接取最近格的 4.5mm，
  内存只有 1/64。插值比堆分辨率有效得多。

- 格中心放在整数坐标上（0, 1, ..., size-1）。
  这样索引就是 round(x)，插值余项就是 x - floor(x)，没有 0.5 偏移的坑。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class Grid:
    """等距网格。

    Attributes:
        size: 每维格数，格中心坐标从 0 到 size-1。
        extent: 每维覆盖 [-extent, +extent]。
    """

    size: int
    extent: float = 1.0

    def __post_init__(self) -> None:
        if self.size < 2:
            raise ValueError("size 至少为 2，否则无法插值")
        if self.extent <= 0:
            raise ValueError("extent 必须为正")

    @property
    def spacing(self) -> float:
        """相邻格中心的实际距离。"""
        return 2.0 * self.extent / (self.size - 1)

    def to_coord(self, delta: np.ndarray) -> np.ndarray:
        """物理坐标 → 格坐标（浮点，可能越界）。"""
        d = np.asarray(delta, dtype=np.float64)
        return (d + self.extent) / (2.0 * self.extent) * (self.size - 1)

    def nearest(self, delta: np.ndarray) -> np.ndarray:
        """最近的格索引（已 clamp 到合法范围）。"""
        idx = np.round(self.to_coord(delta)).astype(int)
        return np.clip(idx, 0, self.size - 1)

    def bilinear(self, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """双线性（任意维张量积）插值的四要素。

        Returns:
            idx0: (n, dim) 下界格索引，已 clamp 到 [0, size-2]。
            frac: (n, dim) 余项，范围 [0, 1]。
            corners: (n, 2**dim, dim) 参与插值的 2**dim 个角点索引。
        """
        d = np.asarray(delta, dtype=np.float64)
        if d.ndim == 1:
            d = d[None, :]
        coord = np.clip(self.to_coord(d), 0.0, self.size - 1.0)
        idx0 = np.clip(np.floor(coord).astype(int), 0, self.size - 2)
        frac = coord - idx0

        offsets = np.array(list(product((0, 1), repeat=d.shape[1])), dtype=int)
        corners = idx0[:, None, :] + offsets[None, :, :]
        return idx0, frac, corners

    def weights(self, frac: np.ndarray) -> np.ndarray:
        """由余项算 2**dim 个角点的插值权重，和为 1。"""
        f = np.asarray(frac, dtype=np.float64)
        if f.ndim == 1:
            f = f[None, :]
        offsets = np.array(list(product((0, 1), repeat=f.shape[1])), dtype=np.float64)
        w = np.prod(np.where(offsets[None, :, :] > 0, f[:, None, :], 1.0 - f[:, None, :]),
                    axis=-1)
        return w
