"""安全层：限幅 + 进展判据。

这一层不是认证用的，是纯算法的「别让它干出蠢事」。

1. 单步转角限幅（`RateLimiter`，已接入 Policy）
   混合之后仍可能有残余跳变（实测 max 10.3°，限幅后 8.0°）。
   限幅是兜底，不是替代混合。注意它是有状态的，换任务要 reset。

2. 进展自检（`ProgressMonitor`，未接入 Policy）
   连续 N 步没更接近目标就判失败。留给「运行期卡死监控」这种场合。
   注意它抓不住「方向偏了」—— 偏差 90° 时距离恒定，距离类指标测不出来。
   真正在用的失败归因做在 episode 级，见 `Policy.end_episode`。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-8)


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """球面插值，把 a 往 b 转 t 比例。任意维度可用。"""
    a = _unit(np.asarray(a, dtype=np.float64))
    b = _unit(np.asarray(b, dtype=np.float64))
    cos = float(np.clip(a @ b, -1.0, 1.0))
    th = np.arccos(cos)
    if th < 1e-6:
        return b.astype(np.float32)
    s = np.sin(th)
    out = (np.sin((1.0 - t) * th) / s) * a + (np.sin(t * th) / s) * b
    return out.astype(np.float32)


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """两个方向向量的夹角（度）。"""
    a = _unit(np.asarray(a, dtype=np.float64))
    b = _unit(np.asarray(b, dtype=np.float64))
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


@dataclass
class RateLimiter:
    """单步转角限幅。

    Attributes:
        max_turn_deg: 相邻两次输出之间允许的最大夹角（度）。
    """

    max_turn_deg: float = 8.0
    _prev: np.ndarray | None = field(default=None, repr=False)

    def reset(self) -> None:
        self._prev = None

    def apply(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        """限幅。

        Returns:
            (限幅后的动作, 是否发生了限幅)
        """
        cur = np.asarray(action, dtype=np.float32)
        if self._prev is None or self.max_turn_deg <= 0:
            self._prev = cur.copy()
            return cur, False

        ang = angle_deg(self._prev, cur)
        if ang <= self.max_turn_deg:
            self._prev = cur.copy()
            return cur, False

        t = self.max_turn_deg / max(ang, 1e-8)
        limited = slerp(self._prev, cur, t)
        self._prev = limited.copy()
        return limited, True


@dataclass
class ProgressMonitor:
    """进展自检：连续若干步都没更接近目标，就判定这条记忆是错的。

    判据换过三次，前两版都错，记下来避免重犯：

    版本一「距离没变小就算失败」：带绕行的动作场里路径是直线的 1.3 倍，
        距离本来就会持平，结果 99.7% 的决策被判失败，记忆全废。
    版本二「比最近最好值差 5% 就算退步」：绕行波动超过 5%，118 条记忆
        被判错 52 条，而它们全是对的。
    版本三（当前）「连续 N 步没有再创新低」：偏差 90° 时距离几乎不变，
        只有「一直不再靠近」才是可靠信号。

    注意偏差 90° 这种错误，距离是测不出来的 —— 垂直方向移动，距离恒定。
    所以这个监视器抓的是「卡住 / 绕圈」，不是「方向偏了」。
    方向偏了要靠双通路的动作分歧来判断，不在这一层。

    Attributes:
        patience: 连续多少步没有更接近目标就判定失败。
    """

    patience: int = 8
    _strikes: int = field(default=0, repr=False)
    _best: float | None = field(default=None, repr=False)

    def reset(self) -> None:
        self._strikes = 0
        self._best = None

    def observe(self, distance: float) -> bool:
        """喂入执行后的剩余距离。

        Returns:
            True 表示应当撤销当前这条记忆。
        """
        d = float(distance)
        if self._best is None or d < self._best - 1e-6:
            self._best = d
            self._strikes = 0
            return False

        self._strikes += 1
        if self._strikes >= self.patience:
            self._strikes = 0
            self._best = None
            return True
        return False
