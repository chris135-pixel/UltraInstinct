"""输出授权门（confidence gate）单元测试。

核心契约：把「联想激活」和「输出授权」分开。
    想得起来一个候选 ≠ 够格把它当确定答案说出来。
verbatim 恒授权；gist/episodic 按 min_confidence 判定。
"""

from __future__ import annotations

import unittest

import numpy as np

from ultra_instinct.encoding import ContentEncoder
from ultra_instinct.forgetting import ForgettingProfile
from ultra_instinct.general_memory import GeneralMemory
from ultra_instinct.hierarchy import GistNode, Hierarchy
from ultra_instinct.quantize import Grid


def drill(mem: GeneralMemory, content: str, response: str,
          n: int, base: int = 0) -> None:
    for k in range(n):
        mem.practice(content, response, base + k, now=base + k)


def gist_cluster(mem: GeneralMemory) -> None:
    """练出一个要旨簇：三条相似输入，其中一条答案冲突，拉低 coherence。"""
    drill(mem, "窗前明月光", "疑似地上霜", 2, 0)
    drill(mem, "床前明月光", "疑似地上霜", 2, 10)
    drill(mem, "月光明窗前", "低头思故乡", 2, 20)
    mem.consolidate()
    mem.compress()


class TestVerbatimAlwaysAuthorized(unittest.TestCase):
    def test_verbatim_confident_even_with_high_gate(self) -> None:
        m = GeneralMemory(min_episodes=2, min_confidence=0.99)
        drill(m, "1+1", "2", 2)
        m.consolidate()
        a = m.query("1+1")
        self.assertEqual(a.source, "verbatim")
        self.assertTrue(a.confident)   # 练熟的不该被任何门误伤

    def test_verbatim_survives_forgetting(self) -> None:
        m = GeneralMemory(min_episodes=2,
                          forgetting=ForgettingProfile())
        drill(m, "1+1", "2", 2)
        m.consolidate()
        a = m.query("1+1", now=3650)
        self.assertTrue(a.confident)   # 逐字满信不受时间衰减影响


class TestGistGate(unittest.TestCase):
    def test_same_gist_splits_by_weight(self) -> None:
        """同一个要旨簇：像得多的过门，边缘/冲突的被拦。

        这是门控存在的核心理由 —— 过去这三条会以同样的姿态返回同一答案。
        """
        m = GeneralMemory(min_episodes=2, coherence_min=0.6,
                          min_confidence=0.6, capacity=64,
                          gradual_decay=False)
        gist_cluster(m)
        self.assertEqual(len(m.gists), 1)
        a_center = m.query("窗前明月光")
        a_edge = m.query("床前明月光")
        self.assertGreater(a_center.weight, a_edge.weight)
        self.assertTrue(a_center.confident)
        self.assertFalse(a_edge.confident)

    def test_gate_none_disables(self) -> None:
        """min_confidence=None 兼容旧行为：一切命中都算授权。"""
        m = GeneralMemory(min_episodes=2, coherence_min=0.6,
                          min_confidence=None, capacity=64,
                          gradual_decay=False)
        gist_cluster(m)
        self.assertTrue(m.query("床前明月光").confident)

    def test_raising_gate_reduces_authorized(self) -> None:
        """门调高，被授权的要旨变少 —— 门是真的可调，不是摆设。"""
        m_low = GeneralMemory(min_episodes=2, coherence_min=0.6,
                              min_confidence=0.5, capacity=64,
                              gradual_decay=False)
        m_high = GeneralMemory(min_episodes=2, coherence_min=0.6,
                               min_confidence=0.7, capacity=64,
                               gradual_decay=False)
        gist_cluster(m_low)
        gist_cluster(m_high)
        self.assertTrue(m_low.query("床前明月光").confident)
        self.assertFalse(m_high.query("床前明月光").confident)

    def test_forgetting_revokes_authorization(self) -> None:
        """随时间遗忘，信任度掉到门下后自动失去授权。"""
        m = GeneralMemory(min_episodes=2, min_confidence=0.6,
                          forgetting=ForgettingProfile(),
                          gradual_decay=False)
        for i, s in enumerate(("窗前明月光", "床前明月光", "月光明窗前")):
            drill(m, s, "疑似地上霜", 2, i * 2)
        m.consolidate()
        m.compress(now=0)
        self.assertTrue(m.query("窗前明月光", now=0).confident)
        a_late = m.query("窗前明月光", now=365)
        self.assertLess(a_late.weight, 0.6)
        self.assertFalse(a_late.confident)

    def test_low_confidence_still_returns_text_for_caller(self) -> None:
        """非严格架构：低置信仍带原文返回，只是标记 False，由调用方裁量。"""
        m = GeneralMemory(min_episodes=2, coherence_min=0.6,
                          min_confidence=0.6, capacity=64,
                          gradual_decay=False)
        gist_cluster(m)
        a = m.query("床前明月光")
        self.assertFalse(a.confident)
        self.assertEqual(a.response, "疑似地上霜")   # 信息没被丢掉


class TestEpisodicGate(unittest.TestCase):
    def test_episodic_is_tentative_by_default(self) -> None:
        """浅记忆未固化，weight 封顶 0.5，默认门 0.6 下不算确定。"""
        m = GeneralMemory(min_episodes=3, min_confidence=0.6)
        drill(m, "床前明月光", "疑是地上霜", 1)
        a = m.query("窗前明月光")
        self.assertEqual(a.source, "episodic")
        self.assertFalse(a.confident)
        self.assertEqual(a.response, "疑是地上霜")   # 但仍有试探价值


class TestHierarchyGate(unittest.TestCase):
    """控制版：低置信要旨动作不被采纳，降级到下一级。"""

    def test_low_confidence_gist_not_used(self) -> None:
        h = Hierarchy(Grid(8, 1.0), 1, 2, min_confidence=0.6)
        center = np.array([0.0, 0.0], dtype=np.float32)
        # coherence=0.3 → 中心处 g_w=0.3 < 0.6
        h.gists = [GistNode(skill_id=0, center=center,
                            action0=np.array([1.0, 0.0], dtype=np.float32),
                            jac=np.zeros((2, 2), dtype=np.float32),
                            radius=1.0, coherence=0.3, curvature=0.0,
                            support=4)]
        act, w, source = h.query(center, 0)
        self.assertNotIn("gist", source)     # 要旨被门挡下
        self.assertEqual(source, "none")

    def test_high_confidence_gist_used(self) -> None:
        h = Hierarchy(Grid(8, 1.0), 1, 2, min_confidence=0.6)
        center = np.array([0.0, 0.0], dtype=np.float32)
        h.gists = [GistNode(skill_id=0, center=center,
                            action0=np.array([1.0, 0.0], dtype=np.float32),
                            jac=np.zeros((2, 2), dtype=np.float32),
                            radius=1.0, coherence=0.9, curvature=0.0,
                            support=4)]
        act, w, source = h.query(center, 0)
        self.assertEqual(source, "gist")
        self.assertGreater(w, 0.6)

    def test_gate_none_disables(self) -> None:
        h = Hierarchy(Grid(8, 1.0), 1, 2, min_confidence=None)
        center = np.array([0.0, 0.0], dtype=np.float32)
        h.gists = [GistNode(skill_id=0, center=center,
                            action0=np.array([1.0, 0.0], dtype=np.float32),
                            jac=np.zeros((2, 2), dtype=np.float32),
                            radius=1.0, coherence=0.3, curvature=0.0,
                            support=4)]
        act, w, source = h.query(center, 0)
        self.assertEqual(source, "gist")     # 不设门时照旧采纳


if __name__ == "__main__":
    unittest.main(verbosity=2)
