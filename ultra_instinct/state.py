"""状态编码：把观测压成网络和查表都能吃的向量。

关键设计（实测依据）：

1. 用相对量 Δ = 目标 − 当前，不用绝对坐标。
   绝对坐标下 20 个目标 × 每点 50 步 ≈ 1000 个 4 维点，彼此几乎不重叠，
   命中率怎么调都上不去。换成 Δ 后所有目标共享同一张表。

2. 必须带 skill_id。
   同一位置、不同意图（伸手取物 vs 绕障避让）的 Δ 完全相同，动作却相反。
   这是信息缺失，不是能力不足。缺这一维，新旧技能必然互相覆盖。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StateSpec:
    """状态向量的规格。

    Attributes:
        dim: Δ 的维度（二维到达任务为 2）。
        n_skills: 技能总数。技能标识按 id/(n_skills-1) 归一化到 [0, 1]。
        scale: 归一化尺度，Δ 除以它再进网络。查表不受影响（表自带 extent）。
    """

    dim: int = 2
    n_skills: int = 1
    scale: float = 1.0

    @property
    def input_dim(self) -> int:
        """网络输入维度 = Δ 维度 + 技能标识。"""
        return self.dim + 1

    def skill_code(self, skill_id: int) -> float:
        """技能标识归一化。单技能时恒为 0。"""
        if self.n_skills <= 1:
            return 0.0
        if not 0 <= skill_id < self.n_skills:
            raise ValueError(f"skill_id {skill_id} 超出范围 [0, {self.n_skills})")
        return skill_id / (self.n_skills - 1)


def encode_state(delta: np.ndarray, skill_id: int, spec: StateSpec) -> np.ndarray:
    """把 Δ 和技能标识拼成网络输入。

    Args:
        delta: (dim,) 或 (n, dim)。相对目标向量。
        skill_id: 技能下标。
        spec: 状态规格。

    Returns:
        (dim+1,) 或 (n, dim+1) 的 float32 数组。
    """
    d = np.asarray(delta, dtype=np.float32)
    scalar = d.ndim == 1
    if scalar:
        d = d[None, :]
    if d.shape[-1] != spec.dim:
        raise ValueError(f"delta 最后一维应为 {spec.dim}，实际 {d.shape[-1]}")

    code = np.float32(spec.skill_code(skill_id))
    out = np.empty((d.shape[0], spec.input_dim), dtype=np.float32)
    out[:, : spec.dim] = d / np.float32(spec.scale)
    out[:, spec.dim] = code
    return out[0] if scalar else out
