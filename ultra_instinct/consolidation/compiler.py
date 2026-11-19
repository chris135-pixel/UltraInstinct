"""编译器：候选条目 → 静态记忆表。"""

from __future__ import annotations

import numpy as np

from ultra_instinct.consolidation.candidate import Entry
from ultra_instinct.memory import LUTMemory
from ultra_instinct.quantize import Grid


def compile_memory(entries: list[Entry], grid: Grid, n_skills: int,
                   act_dim: int = 2) -> LUTMemory:
    """把条目写进网格表并填充空洞。

    一个格子只保留一条记录，所以「去重」是结构自带的，
    不需要额外的覆盖度合并或命中次数淘汰。

    Args:
        entries: 已通过判据的条目。
        grid: 网格规格。
        n_skills: 技能数。
        act_dim: 动作维度。

    Returns:
        可直接查询的 LUTMemory。
    """
    mem = LUTMemory(grid=grid, n_skills=n_skills, act_dim=act_dim)
    for e in entries:
        mem.set_cell(e.skill_id, e.index, e.action)
    return mem.finalize()
