"""通用三级记忆：在「内容空间」里做浅记忆 → 逐字 → 要旨。

这是 hierarchy.py 的通用版。机制完全对应，只换了一件事：

    原版   键 = 二维坐标 (Δx, Δy)，值 = 方向向量，相似度 = 欧氏距离
    通用版 键 = 内容哈希，值 = 任意输出文本，相似度 = 字符 n-gram 余弦

迁移过来的四条约束（都与「内容是坐标还是文字」无关）：

1. 巩固判据是**跨 episode 次数**，不是连续次数。
   同一 episode 里重复问 5 次不算 5 次独立验证。
2. 输出必须稳定才固化。同一输入在不同 episode 给出不同答案时取多数，
   多数不足则拒绝 —— 这是 pattern separation 在内容上的对应。
3. 压缩把相似输入聚成一簇，只保留原型 + 代表输出；
   簇内答案冲突时拒绝合并（否则「窗前明月光」和「床前明月光」
   两种答案会被平均成一个谁都不对的答案）。
4. 压缩是真删除，支撑点没了。再用时靠**再巩固**重新长出逐字条目，
   不是把要旨展开回逐字（fuzzy-trace 理论）。

参考文献见 README「神经科学依据」一节。
"""

from __future__ import annotations

import base64
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ultra_instinct.encoding import ContentEncoder, Encoded
from ultra_instinct.forgetting import ForgettingProfile


# ---------------------------------------------------------------- 数据


@dataclass
class Episode:
    """一条情景痕迹：某次练习里，这个输入给出了这个输出。"""

    enc: Encoded
    response: str
    episode_id: int
    born: float


@dataclass
class VerbatimEntry:
    """逐字条目：这个输入已被验证过，答案稳定，可以直接脱口而出。"""

    enc: Encoded
    response: str
    episodes: set[int] = field(default_factory=set)
    born: float = 0.0
    last_used: float = 0.0
    hits: int = 0

    @property
    def strength(self) -> float:
        return float(len(self.episodes))


@dataclass
class GistNode:
    """要旨节点：一簇相似输入的原型 + 代表输出。

    Attributes:
        center: 簇内输入特征的均值（原型）。
        response: 代表输出，取簇内多数。
        radius: 覆盖半径（到中心的最大相似度阈值）。
        coherence: 簇内答案一致度 0~1。
        support: 代表多少条逐字条目。
        keys: 被合并进来的精确键，命中后可触发再巩固。
        complexity: 被压缩内容的复杂度 0~1（保留多少细节），决定遗忘快慢。
            由 support 推导：合并的逐字越多 → 保留细节越多 → 越先忘。
    """

    center: np.ndarray
    response: str
    radius: float
    coherence: float
    support: int
    keys: set[bytes] = field(default_factory=set)
    born: float = 0.0
    last_used: float = 0.0
    uses: int = 0
    complexity: float = 0.5

    @property
    def n_params(self) -> int:
        """占用量：原型向量 + 一个输出，与簇里多少条无关。"""
        return int(self.center.size + 1)


@dataclass
class Answer:
    """一次查询的结果。

    Attributes:
        response: 输出文本；None 表示完全没命中。
        weight: 信任度 0~1。
        source: verbatim / gist / episodic / none。
        key: 命中的精确键（用于再巩固归因），gist 命中时为 None。
        confident: 是否达到输出授权门槛。调用方若要「宁可不答也不答错」，
            应只采纳 confident=True 的结果；False 表示仅供参考的试探。
            这不是冗余字段 —— weight 是连续值，门槛在内核里，调用方
            不该各自猜阈值。
    """

    response: str | None
    weight: float
    source: str
    key: bytes | None = None
    confident: bool = False


# ---------------------------------------------------------------- 内核


class GeneralMemory:
    """通用三级记忆。

    Args:
        encoder: 内容编码器。
        min_episodes: 跨多少个独立 episode 才固化为逐字。
        stability: 固化所需的多数输出占比。
        capacity: 浅记忆容量上限。
        tau / alpha: 浅记忆幂律衰减参数。
        gist_threshold: 判定「相似」的最小余弦相似度（联想激活门）。
        coherence_min: 允许合并的最低答案一致度（pattern separation 门槛）。
        min_confidence: 输出授权门。要旨/浅记忆命中后信任度必须达到此值
            才标记为 confident=True，否则答案降级为「仅供参考」。这是
            「想得起来」和「够格说出来」的分界，只影响标记与调用方采纳，
            不改变记忆本身；verbatim 始终保持满信。None/0 表示不设门
            （兼容旧行为），默认 0.6。
        forgetting: 被压缩内容的遗忘画像（论文参数）。None 则关闭遗忘。
        complexity_scale: support 归一化到 0~1 的尺度；合并的逐字越多，
            保留细节越多、复杂度越高、按论文机制先被忘掉。
    """

    def __init__(self, encoder: ContentEncoder | None = None,
                 min_episodes: int = 3, stability: float = 0.6,
                 capacity: int = 64, tau: float = 3.0, alpha: float = 1.0,
                 gist_threshold: float = 0.55,
                 coherence_min: float = 0.6,
                 min_confidence: float | None = 0.6,
                 forgetting: ForgettingProfile | None = None,
                 complexity_scale: float = 3.0,
                 gradual_decay: bool = True,
                 detail_threshold: float = 0.5) -> None:
        self.encoder = encoder or ContentEncoder()
        self.min_episodes = int(min_episodes)
        self.stability = float(stability)
        self.capacity = int(capacity)
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.gist_threshold = float(gist_threshold)
        self.coherence_min = float(coherence_min)
        self.min_confidence = (None if min_confidence is None
                               else max(0.0, min(1.0, float(min_confidence))))
        self.forgetting = forgetting
        self.complexity_scale = max(1e-6, float(complexity_scale))
        # 渐进退化（默认开，对应人脑"短期→长期保留完整细节，经过一段
        # 时间不用才渐退成要旨"）：逐字条目随时间连续褪色，用到则保鲜，
        # 褪到阈值以下才在 compress 时简化成要旨。传 gradual_decay=False
        # 可退回旧的"compress 立即一步删"行为（兼容旧调用方）。
        self.gradual_decay = bool(gradual_decay)
        self.detail_threshold = max(0.0, min(1.0, float(detail_threshold)))

        self.episodic: list[Episode] = []
        self.verbatim: dict[bytes, VerbatimEntry] = {}
        self.gists: list[GistNode] = []
        # 巩固计数：key → {episode_id: response}。同一 episode 多次只记最后一次。
        self._pending: dict[bytes, dict[int, str]] = {}
        # 每个 pending key 最近一次练习的时间，固化时作为逐字条目的时间锚点。
        self._pending_time: dict[bytes, float] = {}
        self._feat_cache: dict[bytes, np.ndarray] = {}
        # 原文单独存：浅记忆是 64 条滚动窗口，练几千条后早期条目会被淘汰，
        # 但 _pending 的巩固计数必须完整。原文不能反过来依赖浅记忆。
        self._text_cache: dict[bytes, str] = {}

    def _complexity(self, support: int) -> float:
        """support → 复杂度 0~1：1 条几乎无细节，越多越接近 1。"""
        return float(1.0 - math.exp(-(max(1, int(support)) - 1)
                                     / self.complexity_scale))

    # ---------------------------------------------------------- 写路径

    def _strength(self, ep: Episode, now: float) -> float:
        age = max(0.0, now - ep.born)
        return float((1.0 + age / self.tau) ** (-self.alpha))

    def detail(self, entry: VerbatimEntry, now: float) -> float:
        """逐字条目的当前细节完整度 1→0（渐进退化的核心量）。

        以 last_used 为锚点：距上次使用越久，细节按幂律连续褪去；每次命中
        或再巩固都会刷新 last_used，于是"常用的记得清楚、不用的渐渐模糊"。
        这正好补上原来 born/last_used/hits 只写不读的缺口。

        退化速率用构造时的 tau/alpha（与浅记忆同一套幂律）。关闭渐进退化
        时恒返回 1.0 —— 保证旧行为：逐字永不褪色、compress 立即删。
        """
        if not self.gradual_decay:
            return 1.0
        age = max(0.0, float(now) - float(entry.last_used or entry.born))
        return float((1.0 + age / self.tau) ** (-self.alpha))

    def practice(self, content: str | bytes, response: str | bytes,
                 episode_id: int, now: float = 0.0) -> None:
        """练习一次：写入情景记忆 + 累计巩固计数。此时还不会固化。"""
        enc = self.encoder.encode(content)
        resp = response.decode("utf-8", errors="replace") if isinstance(
            response, bytes) else str(response)

        self.episodic.append(Episode(enc, resp, int(episode_id), float(now)))
        if len(self.episodic) > self.capacity:
            scores = [self._strength(e, now) for e in self.episodic]
            self.episodic.pop(int(np.argmin(scores)))

        self._pending.setdefault(enc.key, {})[int(episode_id)] = resp
        self._pending_time[enc.key] = max(
            float(now), self._pending_time.get(enc.key, float("-inf")))
        self._feat_cache[enc.key] = enc.feat
        self._text_cache[enc.key] = enc.text

    def _promote(self, key: bytes, need: int) -> bool:
        """尝试把单个 key 从 pending 提升为逐字，成功返回 True。

        consolidate 与 reconsolidate 共用这一处判据，避免「固化条件」出现两份
        实现而漂移；更重要的是让 reconsolidate 只处理自己那条，不再顺带把
        别的 pending key 也固化。
        """
        by_ep = self._pending.get(key)
        if by_ep is None or len(by_ep) < need:
            return False
        # 多数输出 + 稳定度检验
        votes: dict[str, int] = {}
        for r in by_ep.values():
            votes[r] = votes.get(r, 0) + 1
        best = max(votes, key=lambda r: votes[r])
        if votes[best] / len(by_ep) < self.stability:
            return False   # 答案不稳定，不固化

        enc = Encoded(key=key, feat=self._feat_cache[key],
                      text=self._text_cache[key])
        born = self._pending_time.get(key, 0.0)
        entry = self.verbatim.get(key)
        if entry is None:
            self.verbatim[key] = VerbatimEntry(
                enc=enc, response=best,
                episodes=set(by_ep.keys()), born=born, last_used=born)
        else:
            entry.episodes.update(by_ep.keys())
            entry.response = best
            # 再巩固后重获细节：用更晚的时间刷新锚点。
            if born > entry.last_used:
                entry.last_used = born
                entry.born = born
        self._pending.pop(key, None)
        self._pending_time.pop(key, None)
        return True

    def consolidate(self, min_episodes: int | None = None) -> int:
        """情景 → 逐字。跨 episode 达标且输出稳定才提升（全部 pending）。"""
        need = self.min_episodes if min_episodes is None else min_episodes
        promoted = 0
        for key in list(self._pending):
            if self._promote(key, need):
                promoted += 1
        self._prune_cache()
        return promoted

    def _prune_cache(self) -> None:
        """缓存只服务 _pending 的固化：逐字条目自带 enc，要旨只存 keys，
        都不依赖这两个缓存。固化/压缩后清掉无引用项，避免随练习无限增长
        （并渗进持久化文件）。
        """
        keep = set(self._pending)
        self._feat_cache = {k: v for k, v in self._feat_cache.items()
                            if k in keep}
        self._text_cache = {k: v for k, v in self._text_cache.items()
                            if k in keep}
        self._pending_time = {k: v for k, v in self._pending_time.items()
                              if k in keep}

    def compress(self, now: float = 0.0,
                 threshold: float | None = None) -> dict:
        """逐字 → 要旨。相似输入聚簇，簇内答案一致才合并，然后真删除逐字条目。

        渐进退化开启时（gradual_decay=True），只有细节完整度已褪到
        detail_threshold 以下的逐字才参与合并；仍清晰的逐字原样保留，
        于是"长期不用 → 细节渐退 → 才简化成要旨"随时间分次发生，
        而不是一次 compress 就把所有细节抹掉。
        """
        thr = self.gist_threshold if threshold is None else threshold
        keys = list(self.verbatim)
        used: set[bytes] = set()
        stats = {"merged": 0, "rejected": 0, "entries_removed": 0,
                 "verbatim_kept": len(keys)}

        for key in keys:
            if key in used:
                continue
            seed = self.verbatim[key]
            cluster = [seed]
            for other_key in keys:
                if other_key in used or other_key == key:
                    continue
                other = self.verbatim[other_key]
                if self.encoder.similarity(seed.enc.feat, other.enc.feat) >= thr:
                    cluster.append(other)

            # 渐进退化：只让已经褪色的逐字进入要旨，鲜活的细节留在逐字层。
            if self.gradual_decay:
                cluster = [e for e in cluster
                           if self.detail(e, now) <= self.detail_threshold]
                if not cluster:
                    continue

            responses = [e.response for e in cluster]
            votes: dict[str, int] = {}
            for r in responses:
                votes[r] = votes.get(r, 0) + 1
            best = max(votes, key=lambda r: votes[r])
            coherence = votes[best] / len(responses)

            if coherence < self.coherence_min:
                stats["rejected"] += 1
                continue   # 相似输入却给不同答案：必须分开记

            center = np.mean(np.stack([e.enc.feat for e in cluster]), axis=0)
            center = center / max(float(np.linalg.norm(center)), 1e-12)
            sims = [self.encoder.similarity(center, e.enc.feat) for e in cluster]
            self.gists.append(GistNode(
                center=center.astype(np.float32),
                response=best,
                radius=float(min(sims)),
                coherence=coherence,
                support=len(cluster),
                keys={e.enc.key for e in cluster},
                born=float(now),
                complexity=self._complexity(len(cluster)),
            ))
            for e in cluster:
                used.add(e.enc.key)
                self.verbatim.pop(e.enc.key, None)
            stats["merged"] += 1
            stats["entries_removed"] += len(cluster)

        stats["verbatim_kept"] = len(self.verbatim)
        self._prune_cache()
        return stats

    def decay(self, now: float = 0.0) -> dict:
        """时间驱动入口：把已褪色的逐字分次沉淀成要旨（compress 的语义别名）。

        与 compress 的区别是语义与默认守卫：decay 代表"随时间自然退化"，
        未开启 gradual_decay 时直接返回 no-op，不会意外抹掉细节。这样调用方
        可以放心地按时间周期调用（如每天一次），由内核决定是否真的退化。
        """
        if not self.gradual_decay:
            return {"merged": 0, "rejected": 0, "entries_removed": 0,
                    "verbatim_kept": len(self.verbatim), "enabled": False}
        stats = self.compress(now=now)
        stats["enabled"] = True
        return stats

    def reconsolidate(self, content: str | bytes, response: str | bytes,
                      episode_id: int, now: float = 0.0) -> bool:
        """再巩固：被要旨命中后拿到的新观测，重新积累成逐字条目。

        不是解压 —— 压缩时支撑点已删，只能靠这次真实观测重新长出来。
        """
        self.practice(content, response, episode_id, now)
        enc = self.encoder.encode(content)
        if enc.key in self.verbatim:
            self.verbatim[enc.key].hits += 1
            self.verbatim[enc.key].last_used = float(now)
            return True
        # 只固化当前 content 这一条，返回它自己的真实结果。
        # 旧实现调用全局 consolidate()：既会顺带固化别的 pending key，
        # 又会因别的 key 被固化而误报 True。
        ok = self._promote(enc.key, self.min_episodes)
        self._prune_cache()
        return ok

    # ---------------------------------------------------------- 读路径

    def _authorized(self, weight: float) -> bool:
        """输出授权：信任度是否够格被当作确定答案采纳。

        min_confidence=None 表示不设门（兼容旧行为），此时一切都算授权。
        verbatim 不走这里 —— 练熟的满信恒授权。
        """
        return self.min_confidence is None or float(weight) >= self.min_confidence

    def query(self, content: str | bytes, now: float = 0.0) -> Answer:
        """三级查询：逐字精确 → 要旨模糊 → 浅记忆表决。

        两级判定：联想激活（能不能想起一个候选）归 radius/threshold 管，
        输出授权（够不够格说出来）归 confident 管。前者放行一个候选，
        不等于后者允许它被当成确定答案 —— 这正是「模糊命中照单全收」
        那个缺陷的修法。
        """
        enc = self.encoder.encode(content)

        entry = self.verbatim.get(enc.key)
        if entry is not None:
            entry.hits += 1
            entry.last_used = float(now)
            return Answer(entry.response, 1.0, "verbatim", enc.key, True)

        best_gist: GistNode | None = None
        best_sim = 0.0
        for g in self.gists:
            sim = self.encoder.similarity(enc.feat, g.center)
            if sim >= g.radius and sim > best_sim:
                best_gist, best_sim = g, sim
        if best_gist is not None:
            best_gist.uses += 1
            best_gist.last_used = float(now)
            weight = best_sim * best_gist.coherence
            if self.forgetting is not None:
                age = max(0.0, float(now) - best_gist.born)
                weight *= self.forgetting.strength(age, best_gist.complexity)
            weight = float(weight)
            return Answer(best_gist.response, weight, "gist", None,
                          self._authorized(weight))

        # 浅记忆表决：像 + 新。weight 封顶 0.5 —— 未固化的东西只算试探，
        # 默认门 0.6 下 confident=False，正好对应「刚学的还不敢拍板」。
        w = np.array([self.encoder.similarity(enc.feat, e.enc.feat)
                      * self._strength(e, now) for e in self.episodic])
        if len(w) and float(w.max()) > self.gist_threshold:
            idx = int(np.argmax(w))
            weight = float(np.clip(w[idx], 0.0, 0.5))
            return Answer(self.episodic[idx].response, weight, "episodic", None,
                          self._authorized(weight))

        return Answer(None, 0.0, "none", None, False)

    # ---------------------------------------------------------- 遗忘

    def forget(self, now: float = 0.0,
               profile: ForgettingProfile | None = None) -> dict:
        """移除已遗忘的要旨：保持率低于阈值的 gist 被真删除。

        按论文机制，复杂度高的（保留细节多的）保持率低，先被移除；
        复杂度低的核心要旨保留更久。删除是真删除，与 compress 一致：
        再用时只能靠 reconsolidate 重新观测长回，不是解压。

        profile 缺省用构造时注入的；两者都 None 时返回空统计（遗忘关闭）。
        """
        pf = profile if profile is not None else self.forgetting
        if pf is None:
            return {"forgotten": 0, "kept": len(self.gists), "enabled": False}

        kept: list[GistNode] = []
        forgotten = 0
        for g in self.gists:
            age = max(0.0, float(now) - g.born)
            if pf.forgotten(age, g.complexity):
                forgotten += 1
            else:
                kept.append(g)
        self.gists = kept
        return {"forgotten": forgotten, "kept": len(kept), "enabled": True}

    def gist_retention(self, now: float = 0.0) -> list[float]:
        """每条 gist 当前的保持率，便于观察「谁先忘」。"""
        if self.forgetting is None:
            return [1.0] * len(self.gists)
        return [self.forgetting.strength(max(0.0, float(now) - g.born),
                                         g.complexity) for g in self.gists]

    # ---------------------------------------------------------- 度量

    @property
    def footprint(self) -> dict[str, int]:
        """各类记忆的占用量（等效参数个数）。"""
        v = sum(1 + e.enc.feat.size for e in self.verbatim.values())
        g = sum(n.n_params for n in self.gists)
        e = len(self.episodic) * (2 + self.encoder.dim)
        return {"verbatim": v, "gist": g, "episodic": e,
                "total": v + g + e}

    # ---------------------------------------------------------- 持久化

    @staticmethod
    def _b64(data: bytes) -> str:
        return base64.b64encode(data).decode("ascii")

    @staticmethod
    def _unb64(text: str) -> bytes:
        return base64.b64decode(text.encode("ascii"))

    @classmethod
    def _enc_json(cls, enc: Encoded) -> dict:
        return {"key": cls._b64(enc.key),
                "feat": cls._b64(np.asarray(enc.feat, dtype=np.float32).tobytes()),
                "text": enc.text}

    @staticmethod
    def _feat_from(d: dict) -> np.ndarray:
        return np.frombuffer(base64.b64decode(d["feat"].encode("ascii")),
                             dtype=np.float32).copy()

    def save(self, path: str | os.PathLike[str]) -> Path:
        """把完整记忆状态写入一个 JSON 文件（bytes/ndarray 用 base64 编码）。

        原子写入：先写同目录临时文件再 os.replace，避免中途崩溃留下半个文件。
        保存的是全部状态（三级记忆 + 巩固计数 + 原文/特征缓存），
        加载后可原样继续练习，不需要重新训练。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "version": 1,
            "config": {
                # encoder 配置必须存：key/feat 都由它生成，恢复时用不同配置
                # 会导致维度错配崩溃或 normalize 不一致而静默查不到。
                "encoder": {
                    "dim": self.encoder.dim,
                    "ngrams": list(self.encoder.ngrams),
                    "normalize": self.encoder.normalize,
                },
                "min_episodes": self.min_episodes,
                "stability": self.stability,
                "capacity": self.capacity,
                "tau": self.tau,
                "alpha": self.alpha,
                "gist_threshold": self.gist_threshold,
                "coherence_min": self.coherence_min,
                "min_confidence": self.min_confidence,
                "complexity_scale": self.complexity_scale,
                "gradual_decay": self.gradual_decay,
                "detail_threshold": self.detail_threshold,
                "forgetting": (None if self.forgetting is None else {
                    "threshold": self.forgetting.threshold,
                    "min_strength": self.forgetting.min_strength,
                }),
            },
            "episodic": [
                {**self._enc_json(e.enc), "response": e.response,
                 "episode_id": e.episode_id, "born": e.born}
                for e in self.episodic
            ],
            "verbatim": [
                {**self._enc_json(v.enc), "response": v.response,
                 "episodes": sorted(v.episodes), "born": v.born,
                 "last_used": v.last_used, "hits": v.hits}
                for v in self.verbatim.values()
            ],
            "gists": [
                {"center": self._b64(np.asarray(g.center, dtype=np.float32).tobytes()),
                 "response": g.response, "radius": g.radius,
                 "coherence": g.coherence, "support": g.support,
                 "keys": [self._b64(k) for k in g.keys],
                 "born": g.born, "last_used": g.last_used, "uses": g.uses,
                 "complexity": g.complexity}
                for g in self.gists
            ],
            "pending": [
                {"key": self._b64(k),
                 "votes": {str(ep): r for ep, r in votes.items()},
                 "time": self._pending_time.get(k, 0.0)}
                for k, votes in self._pending.items()
            ],
            # 缓存必须存全：pending 里的 key 尚未固化、不在 verbatim 中，
            # 恢复后要继续 consolidate() 就得靠这里的特征和原文。
            "feat_cache": [
                {"key": self._b64(k),
                 "feat": self._b64(np.asarray(v, dtype=np.float32).tobytes())}
                for k, v in self._feat_cache.items()
            ],
            "text_cache": [
                {"key": self._b64(k), "text": v}
                for k, v in self._text_cache.items()
            ],
        }

        fd, tmp = tempfile.mkstemp(dir=str(path.parent),
                                   prefix=path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        return path

    @classmethod
    def load(cls, path: str | os.PathLike[str],
             encoder: ContentEncoder | None = None) -> "GeneralMemory":
        """从 save() 写入的文件恢复记忆。文件不存在或损坏时抛异常（不静默降级）。"""
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        if doc.get("version") != 1:
            raise ValueError(f"不支持的记忆文件格式版本：{doc.get('version')!r}")

        cfg = doc["config"]
        # 用文件里的 encoder 配置重建，保证 key/feat 与存盘时一致。
        # 调用方显式传了 encoder 则以它为准；旧文件没有 encoder 字段时
        # 回退到默认 ContentEncoder（与旧行为兼容）。
        if encoder is None and "encoder" in cfg:
            ec = cfg["encoder"]
            encoder = ContentEncoder(
                dim=int(ec["dim"]),
                ngrams=tuple(int(n) for n in ec["ngrams"]),
                normalize=bool(ec["normalize"]),
            )
        pf_cfg = cfg.get("forgetting")
        forgetting = (None if pf_cfg is None else
                      ForgettingProfile(threshold=float(pf_cfg["threshold"]),
                                        min_strength=float(pf_cfg["min_strength"])))
        mem = cls(encoder=encoder,
                  min_episodes=int(cfg["min_episodes"]),
                  stability=float(cfg["stability"]),
                  capacity=int(cfg["capacity"]),
                  tau=float(cfg["tau"]), alpha=float(cfg["alpha"]),
                  gist_threshold=float(cfg["gist_threshold"]),
                  coherence_min=float(cfg["coherence_min"]),
                  min_confidence=cfg["min_confidence"],
                  complexity_scale=float(cfg["complexity_scale"]),
                  gradual_decay=bool(cfg.get("gradual_decay", True)),
                  detail_threshold=float(cfg.get("detail_threshold", 0.5)),
                  forgetting=forgetting)

        mem.episodic = [
            Episode(enc=Encoded(key=cls._unb64(d["key"]), feat=cls._feat_from(d),
                                text=d["text"]),
                    response=d["response"], episode_id=int(d["episode_id"]),
                    born=float(d["born"]))
            for d in doc["episodic"]
        ]
        mem.verbatim = {}
        for d in doc["verbatim"]:
            enc = Encoded(key=cls._unb64(d["key"]), feat=cls._feat_from(d),
                          text=d["text"])
            mem.verbatim[enc.key] = VerbatimEntry(
                enc=enc, response=d["response"],
                episodes={int(e) for e in d["episodes"]},
                born=float(d["born"]), last_used=float(d["last_used"]),
                hits=int(d["hits"]))
        mem.gists = [
            GistNode(center=np.frombuffer(cls._unb64(d["center"]),
                                          dtype=np.float32).copy(),
                     response=d["response"], radius=float(d["radius"]),
                     coherence=float(d["coherence"]), support=int(d["support"]),
                     keys={cls._unb64(k) for k in d["keys"]},
                     born=float(d["born"]), last_used=float(d["last_used"]),
                     uses=int(d["uses"]), complexity=float(d["complexity"]))
            for d in doc["gists"]
        ]
        mem._pending = {
            cls._unb64(d["key"]): {int(ep): r for ep, r in d["votes"].items()}
            for d in doc["pending"]
        }
        mem._pending_time = {
            cls._unb64(d["key"]): float(d.get("time", 0.0))
            for d in doc["pending"]
        }
        mem._feat_cache = {
            cls._unb64(d["key"]): np.frombuffer(
                base64.b64decode(d["feat"].encode("ascii")),
                dtype=np.float32).copy()
            for d in doc["feat_cache"]
        }
        mem._text_cache = {
            cls._unb64(d["key"]): d["text"] for d in doc["text_cache"]
        }
        return mem
