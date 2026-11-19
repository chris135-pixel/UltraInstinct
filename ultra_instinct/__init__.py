"""Ultra Instinct —— 练熟后直接调用的内容/动作记忆系统。

两个 Python 入口 + 一个 HTTP 入口：

    # 通用内容记忆（1+1、诗句、话术）
    from ultra_instinct import GeneralMemory, ContentEncoder
    mem = GeneralMemory(ContentEncoder())
    mem.query("1+1").response

    # 二维控制（机械臂到达）
    from ultra_instinct import Policy
    policy = Policy.load("checkpoints/reach")
    out = policy.act(delta, skill_id=0)

    # OpenAI 兼容的流式 HTTP API：python -m ultra_instinct.api

设计取舍全部来自 benchmarks 的实测结论，见 README。
"""

from ultra_instinct.state import StateSpec, encode_state
from ultra_instinct.quantize import Grid
from ultra_instinct.memory import LUTMemory
from ultra_instinct.net import MLP
from ultra_instinct.policy import Policy, PolicyConfig, ActResult
from ultra_instinct.hierarchy import (
    EpisodicBuffer,
    GistNode,
    Hierarchy,
    compress,
    query_gists,
)
from ultra_instinct.encoding import ContentEncoder, Encoded
from ultra_instinct.semantic import SemanticEncoder
from ultra_instinct.forgetting import ForgettingProfile
from ultra_instinct.general_memory import (
    GeneralMemory,
    Answer,
    Episode,
    VerbatimEntry,
)

__all__ = [
    "StateSpec",
    "encode_state",
    "Grid",
    "LUTMemory",
    "MLP",
    "Policy",
    "PolicyConfig",
    "ActResult",
    "EpisodicBuffer",
    "GistNode",
    "Hierarchy",
    "compress",
    "query_gists",
    "ContentEncoder",
    "Encoded",
    "SemanticEncoder",
    "ForgettingProfile",
    "GeneralMemory",
    "Answer",
    "Episode",
    "VerbatimEntry",
]

__version__ = "0.3.0"
