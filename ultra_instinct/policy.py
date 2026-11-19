"""对外接口：一条状态进去，一个动作出来。

三层路由，全部实测过代价：

    权重 w = 1（四个角点都练熟）   → 直接用手感，不碰网络     64 ns
    权重 w = 0（这一带完全没练）   → 只能现算                 9.2 μs
    0 < w < 1（边界过渡带）        → 两路都算再混合            约 10 μs

实测固化覆盖率 46% 时，网络调用降到 48%；覆盖率 90% 时平均延迟约 1 μs。
边界带占比约 15~22%，这是连续混合的代价。

自检做在 episode 级别，不做在单步级别。原因见 `end_episode`。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from ultra_instinct.blend import blend
from ultra_instinct.memory import LUTMemory
from ultra_instinct.net import MLP
from ultra_instinct.safety import RateLimiter
from ultra_instinct.state import StateSpec, encode_state


@dataclass
class PolicyConfig:
    """运行期参数。默认值都是实测过的推荐值。"""

    n_skills: int = 1
    grid_size: int = 32
    extent: float = 1.0
    act_dim: int = 2
    max_turn_deg: float = 8.0
    unit_action: bool = True
    hit_eps: float = 1e-3


@dataclass
class ActResult:
    """一次决策的完整记录，便于审计。"""

    action: np.ndarray
    weight: float
    source: str
    net_called: bool
    clamped: bool


@dataclass
class Policy:
    """混合策略。

    Attributes:
        net: 慢通路网络。None 表示纯记忆模式。
        memory: 固化记忆表。None 表示纯网络模式（冷启动）。
        config: 运行期参数。
    """

    net: MLP | None
    memory: LUTMemory | None
    config: PolicyConfig = field(default_factory=PolicyConfig)

    def __post_init__(self) -> None:
        self._spec = StateSpec(dim=self.config.act_dim,
                               n_skills=self.config.n_skills)
        self._limiter = RateLimiter(self.config.max_turn_deg)
        self._force_net = {k: False for k in range(self.config.n_skills)}
        # 本 episode 中所有「完全靠手感」的决策，用于失败时归因
        self._episode_hits: dict[int, list[np.ndarray]] = {
            k: [] for k in range(self.config.n_skills)}
        self._last_action = {k: np.zeros(self.config.act_dim, dtype=np.float32)
                             for k in range(self.config.n_skills)}
        self._invalidated = {k: 0 for k in range(self.config.n_skills)}

    # ---------------------------------------------------------------- 推理

    def act(self, delta: np.ndarray, skill_id: int = 0) -> ActResult:
        """给一个相对目标向量，返回动作。

        Args:
            delta: (act_dim,) 相对目标。
            skill_id: 技能下标。
        """
        d = np.atleast_1d(np.asarray(delta, dtype=np.float32))
        n_skills = self.config.n_skills
        if not 0 <= skill_id < n_skills:
            raise ValueError(f"skill_id {skill_id} 超出 [0, {n_skills})")

        source = "net"
        net_called = False
        weight = 0.0

        if self.memory is None:
            # 冷启动：还没有任何记忆，只能现算
            action = self._require_net(d, skill_id)
            net_called = True
        elif self.net is None:
            # 纯记忆模式：查不到就沿上一次的方向滑行。
            # 这是刻意暴露「表不泛化」的代价，生产环境不该这样用，只用于对照实验。
            lut_action, weight = self.memory.query(d, skill_id)
            weight = float(weight)
            if weight >= 1.0 - self.config.hit_eps:
                action = lut_action.astype(np.float32)
                source = "lut"
            else:
                action = self._last_action[skill_id].copy()
                source = "inertia"
        elif self._force_net[skill_id]:
            # 手动熔断：整张表都不可信
            action = self._require_net(d, skill_id)
            source = "fallback"
            net_called = True
        else:
            lut_action, weight = self.memory.query(d, skill_id)
            weight = float(weight)
            if weight >= 1.0 - self.config.hit_eps:
                action = lut_action.astype(np.float32)
                source = "lut"
            elif weight <= self.config.hit_eps:
                action = self._require_net(d, skill_id)
                source = "net"
                net_called = True
            else:
                net_action = self._require_net(d, skill_id)
                action = blend(lut_action, net_action, weight,
                               normalize=self.config.unit_action)[0]
                source = "blend"
                net_called = True

        if source == "lut":
            self._episode_hits[skill_id].append(d.copy())

        self._last_action[skill_id] = action.astype(np.float32)

        if self.config.unit_action:
            n = float(np.linalg.norm(action))
            if n > 1e-8:
                action = (action / n).astype(np.float32)

        action, clamped = self._limiter.apply(action)
        return ActResult(action=action, weight=weight, source=source,
                         net_called=net_called, clamped=clamped)

    def _require_net(self, delta: np.ndarray, skill_id: int) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("没有网络可用")
        return self.net.forward(self._net_input(delta, skill_id))

    def _net_input(self, delta: np.ndarray, skill_id: int) -> np.ndarray:
        return encode_state(delta, skill_id, self._spec)

    # ---------------------------------------------------------------- 自检

    def begin_episode(self, skill_id: int = 0) -> None:
        """开始一次新任务。清空限幅器、惯性方向和本次的手感记录。

        限幅器是有状态的（它就是限制相邻两次输出的转角），
        所以换任务时必须重置，否则上一段轨迹的方向会拖住新任务的第一个动作。
        """
        self._episode_hits[skill_id].clear()
        self._last_action[skill_id] = np.zeros(self.config.act_dim, dtype=np.float32)
        self._limiter.reset()

    def end_episode(self, success: bool, skill_id: int = 0) -> int:
        """结束一次任务。失败就把这次靠手感做的格子标为可疑并撤掉。

        为什么是 episode 级而不是单步级：试过两版单步判据都不成立。
        「距离没变小就算失败」把 99.7% 的决策判死，因为绕行时距离本来就持平。
        「比最好值差 5% 就算退步」把 118 条正确记忆判错 52 条。
        根因是偏差 90° 时距离恒定，过程指标测不出来。

        但「这次任务整体成不成功」是确定的。用它反推哪几步可疑，粒度粗
        但不会误杀 —— 而且归因方向对：手感错了，才会导致这次失败。

        Returns:
            本次撤销的记忆条数。
        """
        hits = self._episode_hits[skill_id]
        n = 0
        if not success and self.memory is not None:
            for d in hits:
                if self.memory.invalidate(d, skill_id) is not None:
                    n += 1
        self._invalidated[skill_id] += n
        hits.clear()
        return n

    def force_net(self, skill_id: int, on: bool = True) -> None:
        """手动熔断：整个技能退回纯网络。留给「整张表都不可信」的场合。"""
        self._force_net[skill_id] = on

    def reset(self, skill_id: int | None = None) -> None:
        """清空限幅器与本次记录。不会恢复已撤销的记忆。"""
        ids = range(self.config.n_skills) if skill_id is None else [skill_id]
        for k in ids:
            self._force_net[k] = False
            self._episode_hits[k].clear()
        self._limiter.reset()

    @property
    def forced_net(self) -> dict[int, bool]:
        return dict(self._force_net)

    @property
    def invalidated_count(self) -> dict[int, int]:
        """各技能累计被撤销的记忆条数，用于观察自检是否在乱杀。"""
        return dict(self._invalidated)

    # ---------------------------------------------------------------- 持久化

    def save(self, dirpath: str) -> None:
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, ensure_ascii=False, indent=2)
        if self.net is not None:
            self.net.save(os.path.join(dirpath, "net.npz"))
        if self.memory is not None:
            self.memory.save(os.path.join(dirpath, "memory.npz"))

    @classmethod
    def load(cls, dirpath: str) -> "Policy":
        with open(os.path.join(dirpath, "config.json"), encoding="utf-8") as f:
            cfg = PolicyConfig(**json.load(f))
        net_path = os.path.join(dirpath, "net.npz")
        mem_path = os.path.join(dirpath, "memory.npz")
        return cls(
            net=MLP.load(net_path) if os.path.exists(net_path) else None,
            memory=LUTMemory.load(mem_path) if os.path.exists(mem_path) else None,
            config=cfg,
        )
