"""分级记忆：浅记忆 → 逐字记忆 → 要旨记忆，以及再巩固回填。

神经科学依据（每条都落到一个可测的量上，不是比喻）：

    浅记忆    EpisodicBuffer   海马：一次性写入、稀疏、容量小、幂律衰减
    逐字      LUTMemory        皮层逐字痕迹：细节完整，可精确查表
    要旨      GistNode         皮层图式：局部仿射模型，只留一阶结构
    再巩固    expand()         提取后变 labile，新样本并入，局部退回逐字

三条约束来自文献，也都被本模块的实测验证过：

1. 压缩损失的**恰好是曲率**（二阶项），一阶结构被仿射模型完整保留。
   所以「丢细节」不是模糊化，是「扔掉弯曲、留下斜率」。
   块半径加倍，误差按 r² 增长 —— 这是可预测的，不是玄学。

2. 不是所有区域都能压。动作场在固化区边界上相差 90°，块内均值互相抵消，
   必须用 coherence 门槛拒绝合并。这就是齿状回的 pattern separation：
   相似但不该混淆的经历，宁可多占地方也不能压成一个。

3. 补细节不是「解压」，是「重新观测」。压缩时支撑点已经删了，还原不了。
   fuzzy-trace 理论的原话是逐字痕迹与要旨痕迹**平行存储**、逐字那条自己衰退，
   不是把要旨展开回逐字。所以 expand() 只在被用到的那一小块上做。

参考文献见 README 的「神经科学依据」一节。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ultra_instinct.memory import LUTMemory
from ultra_instinct.quantize import Grid


# ---------------------------------------------------------------- 浅记忆


@dataclass
class EpisodicTrace:
    """一条情景痕迹。稀疏地记一个具体经历，不做泛化。"""

    skill_id: int
    delta: np.ndarray
    action: np.ndarray
    episode_id: int
    born: float


class EpisodicBuffer:
    """海马式浅记忆：一次性写入、稀疏、容量受限、幂律衰减。

    为什么容量必须小：
        稀疏 + 容量受限是海马区分相似经历（pattern separation）的前提。
        容量一放开它就退化成一张低分辨率表，跟逐字记忆没有区别，
        而低分辨率表恰恰是「把两件不同的事混成一件」的来源。

    为什么衰减用幂律而不是指数：
        Anderson & Schooler (1991) 测出记忆保持量服从幂函数，
        因为环境的「再需要概率」本身就是幂律分布的。
        指数衰减会给出一个特征时间常数，实测对不上；
        幂律没有特征尺度，罕见但关键的记忆不会在某个时刻突然消失。
    """

    def __init__(self, capacity: int = 64, tau: float = 3.0,
                 alpha: float = 1.0, kernel: float = 0.06) -> None:
        self.capacity = capacity
        self.tau = tau
        self.alpha = alpha
        self.kernel = kernel
        self.traces: list[EpisodicTrace] = []

    def __len__(self) -> int:
        return len(self.traces)

    def strength(self, trace: EpisodicTrace, now: float) -> float:
        """幂律保持强度 (1 + age/tau)^-alpha。"""
        age = max(0.0, now - trace.born)
        return float((1.0 + age / self.tau) ** (-self.alpha))

    def write(self, delta: np.ndarray, action: np.ndarray, skill_id: int,
              episode_id: int, now: float = 0.0) -> None:
        """写入一条痕迹。超容量时淘汰最弱的，不是最旧的。"""
        self.traces.append(EpisodicTrace(
            skill_id=skill_id,
            delta=np.asarray(delta, dtype=np.float32).copy(),
            action=np.asarray(action, dtype=np.float32).copy(),
            episode_id=int(episode_id),
            born=float(now),
        ))
        if len(self.traces) > self.capacity:
            scores = [self.strength(t, now) for t in self.traces]
            self.traces.pop(int(np.argmin(scores)))

    def votes(self, delta: np.ndarray, skill_id: int,
              now: float = 0.0) -> tuple[np.ndarray, float]:
        """近邻表决：返回加权动作与总强度。

        权重 = 核相似度 × 幂律强度。两者相乘，等于「像 + 新」。
        """
        d = np.atleast_1d(np.asarray(delta, dtype=np.float32))
        same = [t for t in self.traces if t.skill_id == skill_id]
        if not same:
            return np.zeros(2, dtype=np.float32), 0.0

        dist = np.array([float(np.linalg.norm(t.delta - d)) for t in same])
        sim = np.exp(-(dist / max(self.kernel, 1e-6)) ** 2)
        strg = np.array([self.strength(t, now) for t in same])
        w = sim * strg
        if w.sum() < 1e-8:
            return np.zeros(2, dtype=np.float32), 0.0
        acts = np.stack([t.action for t in same])
        return (acts * w[:, None]).sum(axis=0) / w.sum(), float(w.sum())


# ---------------------------------------------------------------- 要旨记忆


@dataclass
class GistNode:
    """要旨记忆：一个局部仿射模型。

    为什么是仿射而不是「动作均值」：
        均值会丢掉方向随位置变化的信息（斜率），而斜率才是这一带的主旨。
        实测：块内动作夹角 40° 时，均值模型误差约 14°，仿射模型约 1.4°。
        所以「保主旨」的工程含义是「保一阶项」。

    Attributes:
        center: 要旨覆盖区中心（物理坐标）。
        action0: 中心处动作。
        jac: (act_dim, dim) 局部雅可比，就是被保留下来的「主旨」。
        radius: 覆盖半径。
        coherence: 内部一致度 0~1，越低说明块内动作越矛盾，越不可信。
        curvature: 拟合残差 RMS，也就是压缩时丢掉的量。
        support: 代表多少条逐字记忆。
    """

    skill_id: int
    center: np.ndarray
    action0: np.ndarray
    jac: np.ndarray
    radius: float
    coherence: float
    curvature: float
    support: int
    born: float = 0.0
    last_used: float = 0.0
    uses: int = 0

    @property
    def n_params(self) -> int:
        """占用量：动作 + 雅可比，与覆盖多少格无关。"""
        return int(self.action0.size + self.jac.size + self.center.size + 1)

    def evaluate(self, delta: np.ndarray) -> np.ndarray:
        """仿射外推。"""
        d = np.atleast_2d(np.asarray(delta, dtype=np.float64))
        return (self.action0[None, :]
                + (d - self.center[None, :]) @ self.jac.T).astype(np.float32)

    def weight(self, delta: np.ndarray) -> np.ndarray:
        """按到中心的距离给信任度。

        中心处 = coherence（要旨本身就不是满信），越往外越低。
        """
        d = np.atleast_2d(np.asarray(delta, dtype=np.float64))
        dist = np.linalg.norm(d - self.center[None, :], axis=1)
        return (self.coherence
                * np.clip(1.0 - 0.5 * dist / max(self.radius, 1e-6), 0.0, 1.0))

    def covers(self, delta: np.ndarray, slack: float = 1.2) -> np.ndarray:
        d = np.atleast_2d(np.asarray(delta, dtype=np.float64))
        return np.linalg.norm(d - self.center[None, :], axis=1) <= slack * self.radius


def _cell_coord(grid: Grid, index: np.ndarray) -> np.ndarray:
    """格索引 → 物理坐标（格心落在整数坐标上）。"""
    return index / (grid.size - 1) * 2.0 * grid.extent - grid.extent


def _fit_affine(x: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """最小二乘拟合 a ≈ b + J x。

    Returns:
        b: (act_dim,) 截距（中心在原点时即中心动作）。
        J: (act_dim, dim) 雅可比。
        rms: 残差 RMS，即丢掉的高阶成分。
    """
    design = np.concatenate([np.ones((len(x), 1)), x], axis=1)
    coef, *_ = np.linalg.lstsq(design, a, rcond=None)
    pred = design @ coef
    rms = float(np.sqrt(np.mean((a - pred) ** 2)))
    return coef[0], coef[1:].T, rms


# ---------------------------------------------------------------- 压缩


def compress(mem: LUTMemory, block: int = 4, min_support: int = 4,
             coherence_min: float = 0.6) -> tuple[list[GistNode], dict]:
    """把逐字表压成要旨记忆。

    逐字表被就地清空已压的格子 —— 这是真的删除，不是软链接。
    支撑点一旦删掉，细节就无法还原，只能靠再次观测重新获得。
    这是 fuzzy-trace 理论的核心事实，也是本模块 expand() 存在的理由。

    Args:
        mem: 逐字记忆表（会被就地修改）。
        block: 每块的边长（格数）。
        min_support: 一块里至少多少条逐字记忆才值得压。
        coherence_min: 内部一致度门槛。低于此值拒绝合并（pattern separation）。

    Returns:
        (要旨节点列表, 统计信息)
    """
    grid = mem.grid
    nodes: list[GistNode] = []
    stats = {"merged": 0, "rejected": 0, "verbatim_kept": 0,
             "cells_removed": 0, "curvature": []}

    for skill in range(mem.n_skills):
        occ = mem.occupancy[skill] > 0.5
        if not occ.any():
            continue
        idx = np.argwhere(occ)
        groups: dict[tuple[int, ...], list[np.ndarray]] = {}
        for cell in idx:
            key = tuple((cell // block).tolist())
            groups.setdefault(key, []).append(cell)

        for cells in groups.values():
            if len(cells) < min_support:
                stats["verbatim_kept"] += len(cells)
                continue

            cells_arr = np.stack(cells)
            x = _cell_coord(grid, cells_arr.astype(np.float64))
            a = mem.actions[skill][tuple(cells_arr.T)].astype(np.float64)

            center = x.mean(axis=0)
            b, jac, rms = _fit_affine(x - center, a)

            dev = float(np.sqrt(np.mean((a - a.mean(axis=0)) ** 2)))
            if dev < 1e-6:
                coherence = 1.0          # 块内本来就一样，压了也不丢东西
            else:
                coherence = float(np.clip(1.0 - rms / dev, 0.0, 1.0))

            if coherence < coherence_min:
                # 块内动作互相矛盾（典型位置：固化区边界，夹角近 90°）。
                # 平均会得到互相抵消的向量，压出来比不压更危险。
                stats["rejected"] += 1
                stats["verbatim_kept"] += len(cells)
                continue

            # 半径按**块几何**算，不是按支撑点范围。
            # 只取支撑范围会在相邻块之间留缝：格被删掉了，缝里既没要旨也没逐字，
            # 查询直接返回零向量（实测 b=2 时误差 72°、p95 180°，全是这种空响）。
            # 块内只练过一部分格子时，外推到整块是有风险的，
            # 这个风险由 coherence 门槛和 weight 的距离衰减一起兜。
            dim = len(cells_arr[0])
            half_diag = 0.5 * block * grid.spacing * float(np.sqrt(dim))
            support_extent = float(np.max(np.linalg.norm(x - center, axis=1)))
            radius = max(support_extent, half_diag) + 1e-6
            nodes.append(GistNode(
                skill_id=skill,
                center=center.astype(np.float32),
                action0=b.astype(np.float32),
                jac=jac.astype(np.float32),
                radius=radius,
                coherence=coherence,
                curvature=rms,
                support=len(cells),
            ))
            for cell in cells:
                mem.occupancy[skill][tuple(cell)] = 0.0
            stats["merged"] += 1
            stats["cells_removed"] += len(cells)
            stats["curvature"].append(rms)

    return nodes, stats


def query_gists(nodes: list[GistNode], delta: np.ndarray, skill_id: int
                ) -> tuple[np.ndarray | None, float]:
    """在要旨节点里找最合适的一个。

    多个覆盖时取距离最近的 —— 不做加权平均。
    加权平均会把两个不同图式的斜率混起来，得到谁都不像的外推。
    """
    d = np.atleast_1d(np.asarray(delta, dtype=np.float64))
    cands = [n for n in nodes if n.skill_id == skill_id and n.covers(d)[0]]
    if not cands:
        return None, 0.0
    dist = [float(np.linalg.norm(d - n.center)) for n in cands]
    best = cands[int(np.argmin(dist))]
    return best.evaluate(d)[0], float(best.weight(d)[0])


# ---------------------------------------------------------------- 分级记忆


@dataclass
class Hierarchy:
    """三级记忆 + 统一的查询与生命周期管理。

    查询顺序：逐字 → 要旨 → 浅记忆 → None（交给网络）。
    顺序不能反：逐字是精确观测，要旨是它的降级近似，
    只要逐字还在就绝不该用要旨。

    Attributes:
        grid: 网格规格。
        n_skills: 技能数。
        act_dim: 动作维度。
        verbatim: 逐字表。
        episodic: 浅记忆缓冲。
        min_episodes: 浅记忆提升为逐字所需的跨 episode 次数。
        gists: 要旨节点。
    """

    grid: Grid
    n_skills: int = 1
    act_dim: int = 2
    min_episodes: int = 3
    capacity: int = 64
    min_confidence: float | None = 0.6
    """输出授权门。要旨仿射外推的信任度低于此值时不采纳，降级处理。
    None 表示不设门（兼容旧行为）。动作发出去收不回，所以这里用硬门
    而不是像内容版那样只打个标记 —— 两处语义一致：低置信不配当确定答案。"""

    verbatim: LUTMemory = field(init=False)
    episodic: EpisodicBuffer = field(init=False)
    gists: list[GistNode] = field(init=False, default_factory=list)
    _episodes: dict[tuple[int, tuple[int, ...]], set[int]] = field(
        init=False, default_factory=dict)
    _acc: dict[tuple[int, tuple[int, ...]], tuple[np.ndarray, int]] = field(
        init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.verbatim = LUTMemory(self.grid, self.n_skills, self.act_dim)
        self.episodic = EpisodicBuffer(capacity=self.capacity)

    # ------------------------------------------------------------ 写路径

    def practice(self, delta: np.ndarray, action: np.ndarray, skill_id: int,
                 episode_id: int, now: float = 0.0) -> None:
        """练习一次：只写浅记忆。这时还什么都不能固化。

        两个地方都写：
            浅记忆缓冲  最近若干条，用于「还没巩固就先靠它顶一下」
            按格累计    跨 episode 计数 + 动作累加，用于巩固判据
        只靠缓冲会漏 —— 缓冲是 64 条的滚动窗口，练 400 次之后
        早期格子的样本早就被挤掉了（实测 576 个达标格只能提升 32 个）。
        """
        self.episodic.write(delta, action, skill_id, episode_id, now)
        idx = tuple(int(v) for v in self.grid.nearest(np.atleast_1d(delta)))
        key = (skill_id, idx)
        self._episodes.setdefault(key, set()).add(int(episode_id))
        s, n = self._acc.get(key, (np.zeros(self.act_dim, dtype=np.float64), 0))
        self._acc[key] = (s + np.asarray(action, dtype=np.float64), n + 1)

    def consolidate(self, min_episodes: int | None = None,
                    clear: bool = True) -> int:
        """浅记忆 → 逐字记忆（系统巩固）。

        判据是**跨 episode 次数**，不是连续成功次数。
        同一段轨迹里连续几步落在同一格，动作高度相关，
        那不是 N 次独立验证，是 1 次验证重复了 N 遍。
        """
        need = self.min_episodes if min_episodes is None else min_episodes
        n = 0
        for key in list(self._episodes):
            if len(self._episodes[key]) < need:
                continue
            total, cnt = self._acc[key]
            mean = total / max(cnt, 1)
            norm = float(np.linalg.norm(mean))
            if norm > 1e-8:
                mean = mean / norm
            skill_id, idx = key
            self.verbatim.set_cell(skill_id, idx, mean.astype(np.float32))
            n += 1
            if clear:
                self._episodes.pop(key, None)
                self._acc.pop(key, None)
        return n

    def compress_now(self, block: int = 4, min_support: int = 4,
                     coherence_min: float = 0.6) -> dict:
        """逐字 → 要旨（痕迹转化）。"""
        nodes, stats = compress(self.verbatim, block, min_support, coherence_min)
        self.gists.extend(nodes)
        return stats

    def expand(self, delta: np.ndarray, skill_id: int, action: np.ndarray,
               episode_id: int) -> bool:
        """再巩固：在被用到的这一小块上重新长出逐字痕迹。

        不是「解压」。压缩时支撑点已经删掉，无法还原，
        只能靠这次真实执行拿到的新样本重建 —— 所以只是局部、渐进的。
        """
        self.practice(delta, action, skill_id, episode_id)
        idx = tuple(int(v) for v in self.grid.nearest(np.atleast_1d(delta)))
        key = (skill_id, idx)
        if len(self._episodes.get(key, set())) < self.min_episodes:
            return False
        total, cnt = self._acc[key]
        mean = total / max(cnt, 1)
        norm = float(np.linalg.norm(mean))
        self.verbatim.set_cell(skill_id, idx,
                               (mean / norm if norm > 1e-8 else mean).astype(np.float32))
        return True

    # ------------------------------------------------------------ 读路径

    def query(self, delta: np.ndarray, skill_id: int, now: float = 0.0
              ) -> tuple[np.ndarray, float, str]:
        """三级查询。返回 (动作, 信任度, 来源)。"""
        d = np.atleast_1d(np.asarray(delta, dtype=np.float32))

        def authorized(x: float) -> bool:
            return self.min_confidence is None or float(x) >= self.min_confidence

        act, w = self.verbatim.query(d, skill_id)
        if float(w) >= 0.999:
            return act, 1.0, "verbatim"
        if float(w) > 0.0:
            g_act, g_w = query_gists(self.gists, d, skill_id)
            if g_act is not None and authorized(g_w):
                mix = (act * w + g_act * g_w) / max(w + g_w, 1e-8)
                return mix.astype(np.float32), float(w + g_w * 0.5), "verbatim+gist"
            return act, float(w), "verbatim"

        g_act, g_w = query_gists(self.gists, d, skill_id)
        if g_act is not None and authorized(g_w):
            return g_act, float(g_w), "gist"

        e_act, e_w = self.episodic.votes(d, skill_id, now)
        if e_w > 1e-6:
            return e_act.astype(np.float32), float(min(e_w, 0.5)), "episodic"
        return np.zeros(self.act_dim, dtype=np.float32), 0.0, "none"

    def invalidate(self, delta: np.ndarray, skill_id: int) -> bool:
        """撤销：优先撤逐字格；该格已被压缩则把对应要旨降权。"""
        idx = tuple(int(v) for v in self.grid.nearest(np.atleast_1d(delta)))
        if self.verbatim.occupancy[(skill_id,) + idx] > 0.5:
            self.verbatim.occupancy[(skill_id,) + idx] = 0.0
            return True
        d = np.asarray(delta, dtype=np.float64)
        for node in self.gists:
            if node.skill_id == skill_id and node.covers(d)[0]:
                node.coherence *= 0.5
                return True
        return False

    # ------------------------------------------------------------ 度量

    @property
    def footprint(self) -> dict[str, int]:
        """各类记忆占用的浮点参数量。"""
        v = int((self.verbatim.occupancy > 0.5).sum()) * (self.act_dim + 2)
        g = int(sum(n.n_params for n in self.gists))
        e = int(len(self.episodic) * (self.act_dim * 2 + 2))
        return {"verbatim": v, "gist": g, "episodic": e, "total": v + g + e}
