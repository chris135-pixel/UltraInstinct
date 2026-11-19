"""候选池：什么动作才配进记忆表。

判据为什么要「跨 episode」而不是「连续 N 次」：

    同一段轨迹里连续 5 步都落在同一格，动作高度相关，
    那不是 5 次独立验证，是 1 次验证重复了 5 遍。
    要求跨 5 个不同 episode，才是 5 次独立复现。

门槛为什么是 5 而不是 30：

    一个状态簇平均被访问 8 次，凑够 30 次连续成功的概率趋近 0，
    覆盖率会直接归零。实测：
        N=5   覆盖率 90%，无自检时每百次用错 14.3 次
        N=30  覆盖率  0%，无自检时每百次用错  3.1 次
    加上「连续 2 次没进展就回退」之后：
        N=5 降到 1.88 次，N=30 降到 1.49 次
    差 26%，但覆盖率差 90 个百分点。所以选 5，可靠性交给自检补。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ultra_instinct.quantize import Grid


@dataclass
class Entry:
    """一条待固化或已固化的记忆。"""

    skill_id: int
    index: tuple[int, ...]
    action: np.ndarray
    n_episodes: int
    spread: float


@dataclass
class _Cell:
    episodes: set[int] = field(default_factory=set)
    actions: list[np.ndarray] = field(default_factory=list)


class CandidatePool:
    """按格聚合练习结果，够条件的提升为记忆条目。

    Attributes:
        grid: 网格规格。
        n_skills: 技能数。
        min_episodes: 需要多少个不同 episode 成功。
        max_spread: 动作一致性上限（相对离散度），超了说明这一格多解，不能固化。
    """

    def __init__(self, grid: Grid, n_skills: int = 1,
                 min_episodes: int = 5, max_spread: float = 0.25) -> None:
        self.grid = grid
        self.n_skills = n_skills
        self.min_episodes = min_episodes
        self.max_spread = max_spread
        self._cells: dict[tuple[int, tuple[int, ...]], _Cell] = {}

    def record(self, delta: np.ndarray, skill_id: int, action: np.ndarray,
               episode_id: int) -> None:
        """记录一次成功执行。

        Args:
            delta: 执行时的相对目标向量。
            skill_id: 技能下标。
            action: 实际输出的动作。
            episode_id: 这次练习属于哪个 episode（用于跨 episode 计数）。
        """
        idx = tuple(int(v) for v in self.grid.nearest(np.atleast_1d(delta)))
        key = (skill_id, idx)
        cell = self._cells.setdefault(key, _Cell())
        cell.episodes.add(int(episode_id))
        cell.actions.append(np.asarray(action, dtype=np.float32))

    def _spread(self, actions: list[np.ndarray]) -> float:
        """动作相对离散度：平均偏离均值向量的比例。"""
        a = np.stack(actions)
        mean = a.mean(axis=0)
        scale = float(np.linalg.norm(mean))
        if scale < 1e-8:
            return float("inf")
        return float(np.mean(np.linalg.norm(a - mean, axis=1)) / scale)

    def promoted(self) -> list[Entry]:
        """返回已达到固化条件的条目。"""
        out: list[Entry] = []
        for (skill_id, idx), cell in self._cells.items():
            if len(cell.episodes) < self.min_episodes:
                continue
            spread = self._spread(cell.actions)
            if spread > self.max_spread:
                continue
            mean = np.stack(cell.actions).mean(axis=0)
            norm = float(np.linalg.norm(mean))
            if norm > 1e-8:
                mean = mean / norm
            out.append(Entry(skill_id=skill_id, index=idx,
                             action=mean.astype(np.float32),
                             n_episodes=len(cell.episodes), spread=spread))
        return out

    def stats(self) -> dict[str, float]:
        """候选池现状，便于观察练到什么程度。"""
        n_cells = len(self._cells)
        n_ready = len(self.promoted())
        return {
            "cells_touched": n_cells,
            "cells_ready": n_ready,
            "ready_ratio": n_ready / n_cells if n_cells else 0.0,
        }
