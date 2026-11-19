"""通用内容记忆单元测试。零额外依赖，标准库 unittest。"""

from __future__ import annotations

import unittest

import numpy as np

from ultra_instinct.encoding import ContentEncoder
from ultra_instinct.general_memory import GeneralMemory


def drill(mem: GeneralMemory, content: str, response: str,
          n: int, base: int = 0) -> None:
    for k in range(n):
        mem.practice(content, response, base + k, now=base + k)


class TestEncoding(unittest.TestCase):
    def test_key_stable_across_instances(self) -> None:
        a = ContentEncoder()
        b = ContentEncoder()
        self.assertEqual(a.key_of("1+1"), b.key_of("1+1"))

    def test_normalization(self) -> None:
        e = ContentEncoder()
        self.assertEqual(e.clean("  １＋１  "), e.clean("1+1"))

    def test_similarity_near_miss_vs_unrelated(self) -> None:
        e = ContentEncoder()
        near = e.similarity(e.features("窗前明月光"), e.features("床前明月光"))
        far = e.similarity(e.features("1+1"), e.features("床前明月光"))
        self.assertGreater(near, 0.5)
        self.assertLess(far, 0.1)

    def test_feature_l2_normalized(self) -> None:
        e = ContentEncoder()
        v = e.features("任意内容测试")
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)

    def test_empty_content_safe(self) -> None:
        e = ContentEncoder()
        self.assertEqual(float(np.linalg.norm(e.features(""))), 0.0)


class TestConsolidation(unittest.TestCase):
    def test_not_solid_before_threshold(self) -> None:
        m = GeneralMemory(min_episodes=3)
        drill(m, "1+1", "2", 2)
        m.consolidate()
        self.assertEqual(len(m.verbatim), 0)

    def test_solid_at_threshold(self) -> None:
        m = GeneralMemory(min_episodes=3)
        drill(m, "1+1", "2", 3)
        promoted = m.consolidate()
        self.assertEqual(promoted, 1)
        self.assertEqual(m.query("1+1").source, "verbatim")

    def test_same_episode_repeat_not_counted(self) -> None:
        m = GeneralMemory(min_episodes=3)
        for _ in range(10):
            m.practice("1+1", "2", episode_id=0)  # 全是同一 episode
        m.consolidate()
        self.assertEqual(len(m.verbatim), 0)

    def test_unstable_answer_not_solid(self) -> None:
        m = GeneralMemory(min_episodes=3, stability=0.8)
        for ep, ans in enumerate(("2", "2", "3")):
            m.practice("1+1", ans, episode_id=ep)
        m.consolidate()
        self.assertEqual(len(m.verbatim), 0)

    def test_drill_does_not_depend_on_episodic(self) -> None:
        """练的量远超浅记忆容量时，巩固仍要拿到原文（回归 bug）。"""
        m = GeneralMemory(min_episodes=2, capacity=4)
        for i in range(50):
            drill(m, f"问题{i}", f"答案{i}", 2, base=i * 2)
        n = m.consolidate()
        self.assertEqual(n, 50)
        self.assertEqual(m.query("问题49").response, "答案49")


class TestQuery(unittest.TestCase):
    def test_unknown_not_answered(self) -> None:
        m = GeneralMemory()
        a = m.query("从没见过的问题")
        self.assertIsNone(a.response)
        self.assertEqual(a.source, "none")

    def test_episodic_fallback_before_solid(self) -> None:
        m = GeneralMemory(min_episodes=3)
        drill(m, "床前明月光", "疑是地上霜", 1)
        a = m.query("窗前明月光")  # 错字
        self.assertEqual(a.response, "疑是地上霜")
        self.assertEqual(a.source, "episodic")


class TestCompression(unittest.TestCase):
    def test_similar_same_answer_merged(self) -> None:
        m = GeneralMemory(min_episodes=2)
        for i, s in enumerate(("窗前明月光", "床前明月光", "月光明窗前")):
            drill(m, s, "疑似地上霜", 2, base=i * 2)
        m.consolidate()
        before = m.footprint["total"]
        st = m.compress()
        self.assertGreater(st["merged"], 0)
        self.assertEqual(len(m.gists), 1)
        self.assertLess(m.footprint["total"], before)
        self.assertEqual(m.query("窗前明月光").response, "疑似地上霜")

    def test_conflicting_answer_rejected(self) -> None:
        """相似输入不同答案：必须分开，不能平均成一个错答案。"""
        m = GeneralMemory(min_episodes=2, coherence_min=0.6)
        drill(m, "窗前明月光", "疑似地上霜", 2, 0)
        drill(m, "床前明月光", "低头思故乡", 2, 10)
        m.consolidate()
        st = m.compress()
        self.assertEqual(st["rejected"], 2)
        self.assertEqual(len(m.gists), 0)
        self.assertEqual(m.query("窗前明月光").response, "疑似地上霜")
        self.assertEqual(m.query("床前明月光").response, "低头思故乡")


class TestReconsolidation(unittest.TestCase):
    def test_verbatim_regrows_after_compression(self) -> None:
        m = GeneralMemory(min_episodes=2)
        for i, s in enumerate(("窗前明月光", "床前明月光")):
            drill(m, s, "疑似地上霜", 2, base=i * 2)
        m.consolidate()
        m.compress()
        self.assertEqual(len(m.verbatim), 0)
        self.assertEqual(m.query("窗前明月光").source, "gist")
        m.reconsolidate("窗前明月光", "疑似地上霜", 100, now=5)
        m.reconsolidate("窗前明月光", "疑似地上霜", 101, now=6)
        self.assertEqual(m.query("窗前明月光", now=7).source, "verbatim")


class TestCapacity(unittest.TestCase):
    def test_thousand_entries_no_collision(self) -> None:
        m = GeneralMemory(min_episodes=2, capacity=16)
        N = 1000
        for i in range(N):
            drill(m, f"问题{i}", f"答案{i}", 2, base=i * 2)
        m.consolidate()
        self.assertEqual(len(m.verbatim), N)
        for i in range(0, N, 97):
            self.assertEqual(m.query(f"问题{i}").response, f"答案{i}")

    def test_new_learning_keeps_old(self) -> None:
        m = GeneralMemory(min_episodes=2)
        drill(m, "1+1", "2", 2, 0)
        m.consolidate()
        for i in range(200):
            drill(m, f"新题{i}", f"新答{i}", 2, base=10 + i * 2)
        m.consolidate()
        self.assertEqual(m.query("1+1").response, "2")
        self.assertEqual(m.query("新题199").response, "新答199")


if __name__ == "__main__":
    unittest.main(verbosity=2)
