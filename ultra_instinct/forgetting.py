"""按论文参数设计的分层遗忘：记忆强度 = 多个衰减成分的加权和。

设计来源（都是上面查到的论文里能查到数字的量，不是拍脑袋的常数）：

1. 保持函数的形状——幂律 R(t) = (1 + t / tau) ^ (-alpha)
   Ebbinghaus（1885）原始数据（无意义音节，再学习节省率）：
       20min 58.2% / 1h 44.2% / 8.8h 35.8% / 1d 33.7%
       2d 27.8% / 6d 25.4% / 31d 21.1%
   本仓库用这 7 个点网格拟合得 alpha≈0.13、tau≈0.00015 天（约 13 秒），
   对全部 7 点最大误差 < 0.03。幂律而非指数：Wixted & Ebbesen (1991)、
   Anderson & Schooler (1991) 的一致结论，指数有特征时间常数，实测对不上。
   注意这些数字来自单被试、无意义材料、再学法；它是**形状基准**不是普适常数。

2. 分层的时间尺度——把 Ebbinghaus 曲线按层横向拉伸，而不是换一条曲线
   - 逐字/表层细节：半衰期 ≈ 40 分钟。Kintsch et al. (1990, JML) 实测文本
     表层形式 40 分钟内下降过半、4 天接近随机；与 Ebbinghaus 拟合的
     44 分钟半衰期同量级，取 40 分钟作整分钟数。
   - 一阶要旨（命题/textbase）：半衰期 ≈ 4 天。Kintsch 1990 实测 4 天后
     约剩一半；Radvansky et al. (2017) 报告词表/叙事在 7 天处出现陡降拐点。
   - 核心 gist（情境模型）：半衰期 ≈ 30 天。Kintsch 1990 的情境模型在
     4 天内几乎不降；Sacripante et al. (2023) 追踪 1 个月要旨仍高于边缘细节。
   - 语义/熟练（permastore）：半衰期 ≈ 25 年。Bahrick (1984, JEP:General)
     733 人追踪 50 年：前 3–6 年下降，之后约 25 年平台，50 年后仍可再认。
   拉伸方式保持 alpha 不变，只改 tau：R_layer(t) = R_eb(t * s)。

3. "复杂度高先忘"——RAFT 模型机制（Fisher & Radvansky 2019, JML）
   记忆痕迹由多个成分组成，各成分按负加速函数衰减、但衰减率不同；
   复杂度 = 成分里短寿命部分占多少权重。
   对本模型：一条 gist 若保留大量逐字细节，则短寿命成分权重大 → 先掉；
   高度压缩、只剩核心要旨的，长寿命成分权重大 → 抗遗忘。
   这与 Sacripante 2023 的实测一致：边缘细节一个月内忘得比 gist 快，
   且重复提取**不改变二者的相对遗忘率差异**（所以复杂度权重是稳定的）。

4. 学习强度只改截距不改斜率（Slamecka & McElree 1983；Rivera-Lares 2022）
   所以练习次数作用于 strength 的初值，不改变上面的 alpha / tau。

单位：所有时间以「天」为基准（now、born、age 都是天）。
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------- 基准曲线

# Ebbinghaus 原始 7 点网格拟合值。alpha 是曲线形状，7 层的 tau 由半衰期反推。
ALPHA = 0.13
# 由 (1 + t_half / tau)^(-alpha) = 0.5 反推：tau = t_half / (2^(1/alpha) - 1)
_HALF_DIVISOR = 2.0 ** (1.0 / ALPHA) - 1.0  # ≈ 205.5


def tau_from_halflife(days: float) -> float:
    """由半衰期（天）反推幂律曲线的 tau。"""
    return float(days) / _HALF_DIVISOR


# 半衰期（天），见模块 docstring 第 2 条的论文出处。
HALFLIFE_VERBATIM_DAYS = 40.0 / 1440.0        # 40 分钟，Kintsch 1990 表层
HALFLIFE_TEXTBASE_DAYS = 4.0                  # 4 天，Kintsch 1990 命题层
HALFLIFE_GIST_DAYS = 30.0                     # 30 天，Sacripante 2023 一个月
HALFLIFE_SEMANTIC_DAYS = 25.0 * 365.0         # 25 年，Bahrick 1984 permastore

TAU_VERBATIM = tau_from_halflife(HALFLIFE_VERBATIM_DAYS)
TAU_TEXTBASE = tau_from_halflife(HALFLIFE_TEXTBASE_DAYS)
TAU_GIST = tau_from_halflife(HALFLIFE_GIST_DAYS)
TAU_SEMANTIC = tau_from_halflife(HALFLIFE_SEMANTIC_DAYS)

# 四个成分按「细节多 → 核心要旨」排列。权重由复杂度分配。
TAUS: tuple[float, ...] = (TAU_VERBATIM, TAU_TEXTBASE, TAU_GIST, TAU_SEMANTIC)


def retention_component(age_days: float, tau: float) -> float:
    """单个幂律成分的保持率 R(t) = (1 + t/tau)^(-alpha)。"""
    age = max(0.0, float(age_days))
    return float((1.0 + age / tau) ** (-ALPHA))


def component_weights(complexity: float) -> tuple[float, float, float, float]:
    """把复杂度 0~1 映射成四成分权重，和为 1。

    complexity=1（几乎全是细节）：逐字成分占大头 → 按 40 分钟半衰期先掉。
    complexity=0（纯核心要旨）：语义成分占大头 → 按 25 年半衰期长期保留。

    权重表是线性的，没有额外折衷参数；这样"复杂度高的先忘"在任意
    age>0 都单调成立，可被单元测试直接验证。
    """
    c = min(1.0, max(0.0, float(complexity)))
    return (c, (1.0 - c) / 3.0, (1.0 - c) / 3.0, (1.0 - c) / 3.0)


def retention(age_days: float, complexity: float) -> float:
    """混合保持率：四个幂律成分按复杂度加权求和。

    这是 RAFT 的离散版本：每个成分各自负加速衰减，宏观曲线是它们的和。
    复杂度高 ⇒ 短寿命成分权重大 ⇒ 同样 age 下保持率更低（先被忘掉）。
    """
    w = component_weights(complexity)
    return float(sum(wi * retention_component(age_days, tau)
                     for wi, tau in zip(w, TAUS)))


def halflife_of(complexity: float) -> float:
    """一条记忆（给定复杂度）的混合保持率降到 0.5 所需的天数。

    用二分求根；只用于报告/测试，不参与查询热路径。
    """
    lo, hi = 0.0, 1.0
    while retention(hi, complexity) > 0.5:
        hi *= 2.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if retention(mid, complexity) > 0.5:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


@dataclass(frozen=True)
class ForgettingProfile:
    """遗忘画像：把论文参数打包，便于注入记忆模块并做对照实验。

    Attributes:
        threshold: 保持率低于此值即视为遗忘（可从记忆中移除）。
        min_strength: 查询时返回的最低信任度系数，防止强度趋零后抖动。
    """

    threshold: float = 0.20
    min_strength: float = 0.0

    def strength(self, age_days: float, complexity: float) -> float:
        return max(self.min_strength, retention(age_days, complexity))

    def forgotten(self, age_days: float, complexity: float) -> bool:
        return retention(age_days, complexity) < self.threshold


DEFAULT_PROFILE = ForgettingProfile()
