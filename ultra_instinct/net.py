"""慢通路：纯 numpy 的 MLP 前向。

为什么不用 torch 做推理：

    实测本机单步前向        torch eager   55.5 μs（预分配）/ 81.9 μs（每步新建）
    同结构手写 numpy                       9.2 μs
    差 6~9 倍，差的全是框架调度开销，不是算力。

所以训练用 torch，推理零 torch 依赖。权重导出成 npz，前向只有几次矩阵乘。

网络规模选择也有实测依据：MAC 数涨 1657 倍，延迟只涨 6.9 倍
（固定开销约 8.4 μs，边际成本 0.1~0.2 ns/MAC）。
所以不要为了速度缩模型，那是在牺牲能力换噪音。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MLP:
    """三层全连接网络，tanh 激活。

    Attributes:
        w1, b1, w2, b2, w3, b3: 权重，线性层权重已转置成 (in, out) 便于直接矩阵乘。
        out_tanh: 输出层是否过 tanh（方向型动作需要，位移型不需要）。
    """

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray
    out_tanh: bool = True

    @property
    def input_dim(self) -> int:
        return self.w1.shape[0]

    @property
    def output_dim(self) -> int:
        return self.w3.shape[1]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向。

        Args:
            x: (input_dim,) 或 (n, input_dim)。

        Returns:
            同批次形状的 (.., output_dim)。
        """
        a = np.asarray(x, dtype=np.float32)
        scalar = a.ndim == 1
        if scalar:
            a = a[None, :]
        if a.shape[-1] != self.input_dim:
            raise ValueError(f"输入维度应为 {self.input_dim}，实际 {a.shape[-1]}")

        h = np.tanh(a @ self.w1 + self.b1)
        h = np.tanh(h @ self.w2 + self.b2)
        y = h @ self.w3 + self.b3
        if self.out_tanh:
            y = np.tanh(y)
        return y[0] if scalar else y

    # ---------------------------------------------------------------- 持久化

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2,
            w3=self.w3, b3=self.b3, out_tanh=np.bool_(self.out_tanh),
        )

    @classmethod
    def load(cls, path: str) -> "MLP":
        with np.load(path) as z:
            return cls(
                w1=z["w1"].astype(np.float32), b1=z["b1"].astype(np.float32),
                w2=z["w2"].astype(np.float32), b2=z["b2"].astype(np.float32),
                w3=z["w3"].astype(np.float32), b3=z["b3"].astype(np.float32),
                out_tanh=bool(z["out_tanh"]),
            )
