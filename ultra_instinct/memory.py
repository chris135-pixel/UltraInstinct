"""动作记忆表：每个技能一张网格表，双线性查询。

设计依据（实测）：

- 一个技能一张表，不是把所有技能塞进同一张网格的额外维度。
  技能维度放进网格会让格数翻 n 倍，而技能之间本来就不共享邻域。

- 查询返回两部分：动作值 + 固化权重。
  权重是「这块区域练熟了多少」，由固化与否做插值得到。
  输出时用 weight 在两路之间加权，边界处自然连续。

- 动作只由「已固化的角点」加权平均得出，未固化的格子不参与。
  第一版是先把未固化格填成邻近值再插值的，结果外推值污染了混合结果：
  delta 朝左上，输出却朝右上。填一个假值不如根本不用它。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ultra_instinct.quantize import Grid


@dataclass
class LUTMemory:
    """固化动作表。

    Attributes:
        grid: 网格规格。
        n_skills: 技能数。
        act_dim: 动作维度。
    """

    grid: Grid
    n_skills: int
    act_dim: int = 2

    def __post_init__(self) -> None:
        shape = (self.n_skills,) + (self.grid.size,) * 2
        self.actions = np.zeros(shape + (self.act_dim,), dtype=np.float32)
        self.occupancy = np.zeros(shape, dtype=np.float32)
        self.hits = np.zeros(shape, dtype=np.int32)

    # ---------------------------------------------------------------- 写入

    def set_cell(self, skill_id: int, index: tuple[int, ...], action: np.ndarray,
                 occupied: bool = True) -> None:
        """直接写一个格子（编译器用）。"""
        self.actions[(skill_id,) + tuple(index)] = np.asarray(action, dtype=np.float32)
        self.occupancy[(skill_id,) + tuple(index)] = 1.0 if occupied else 0.0

    def finalize(self) -> "LUTMemory":
        """编译收尾。保留为显式步骤，便于编译器表达「写完了」这个语义。"""
        return self

    @property
    def coverage(self) -> np.ndarray:
        """每个技能的固化覆盖率。"""
        return self.occupancy.reshape(self.n_skills, -1).mean(axis=1)

    # ---------------------------------------------------------------- 查询

    def query(self, delta: np.ndarray, skill_id: int) -> tuple[np.ndarray, np.ndarray]:
        """查表。

        Args:
            delta: (dim,) 或 (n, dim) 相对目标向量。
            skill_id: 技能下标。

        Returns:
            action: (n, act_dim) 已固化角点的加权平均。
            weight: (n,) 固化权重，0 表示这一带完全没练过，1 表示完全练熟。
        """
        scalar = np.asarray(delta).ndim == 1
        d = np.atleast_2d(np.asarray(delta, dtype=np.float64))
        _, frac, corners = self.grid.bilinear(d)
        w = self.grid.weights(frac)

        table = self.actions[skill_id]
        flat = table.reshape(-1, self.act_dim)
        occ = self.occupancy[skill_id].reshape(-1)

        flat_idx = np.ravel_multi_index(corners.transpose(2, 0, 1), table.shape[:2])
        vals = flat[flat_idx]                       # (n, k, act_dim)
        mask = occ[flat_idx]                        # (n, k) 已固化角点

        wm = w * mask                               # 未固化角点权重归零
        total = wm.sum(axis=1)
        action = (vals * wm[..., None]).sum(axis=1) / np.maximum(total, 1e-8)[:, None]

        if scalar:
            return action[0].astype(np.float32), np.float32(total[0])
        return action.astype(np.float32), total.astype(np.float32)

    # ---------------------------------------------------------------- 纠错

    def invalidate(self, delta: np.ndarray, skill_id: int) -> tuple[int, ...] | None:
        """撤销一条记忆：把落到的那个格子标成未固化。

        撤销粒度是单个格子，不是整个技能。一条记忆错了不代表别的格也错，
        整表退回纯网络等于把练熟的功夫全扔了（实测过一次：自检一触发，
        网络调用率立刻回到 99.7%，记忆完全失效）。

        Returns:
            被撤销的格索引；该格本来就没固化则返回 None（避免重复计数）。
        """
        idx = tuple(int(v) for v in self.grid.nearest(np.atleast_1d(delta)))
        key = (skill_id,) + idx
        if self.occupancy[key] <= 0.0:
            return None
        self.occupancy[key] = 0.0
        return idx

    # ---------------------------------------------------------------- 持久化

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            actions=self.actions,
            occupancy=self.occupancy,
            hits=self.hits,
            size=np.int32(self.grid.size),
            extent=np.float64(self.grid.extent),
            n_skills=np.int32(self.n_skills),
            act_dim=np.int32(self.act_dim),
        )

    @classmethod
    def load(cls, path: str) -> "LUTMemory":
        with np.load(path) as z:
            mem = cls(
                grid=Grid(int(z["size"]), float(z["extent"])),
                n_skills=int(z["n_skills"]),
                act_dim=int(z["act_dim"]),
            )
            mem.actions = z["actions"].astype(np.float32)
            mem.occupancy = z["occupancy"].astype(np.float32)
            mem.hits = z["hits"].astype(np.int32)
        return mem.finalize()
