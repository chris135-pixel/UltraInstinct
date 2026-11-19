"""按论文参数的分层遗忘单元测试。零额外依赖，标准库 unittest。"""

from __future__ import annotations

import unittest

import ultra_instinct.forgetting as F
from ultra_instinct.forgetting import ForgettingProfile
from ultra_instinct.general_memory import GeneralMemory


def drill(mem: GeneralMemory, content: str, response: str,
          n: int, base: int = 0) -> None:
    for k in range(n):
        mem.practice(content, response, base + k, now=base + k)


class TestPaperParameters(unittest.TestCase):
    """数值必须落在论文报告的量级上，防止参数被随手改坏。"""

    def test_ebbinghaus_seven_points(self) -> None:
        """逐字成分的保持率复现 Ebbinghaus 原始 7 点（误差 < 0.05）。"""
        pts = [(20 / 1440, 0.582), (1 / 24, 0.442), (9 / 24, 0.358),
               (1, 0.337), (2, 0.278), (6, 0.254), (31, 0.211)]
        for t, observed in pts:
            pred = F.retention_component(t, F.TAU_VERBATIM)
            self.assertLess(abs(pred - observed), 0.05, f"t={t}")

    def test_four_layer_halflives(self) -> None:
        """四成分半衰期 = 40分钟 / 4天 / 30天 / 25年。"""
        expected = [40 / 1440, 4.0, 30.0, 25.0 * 365]
        for tau, exp in zip(F.TAUS, expected):
            self.assertAlmostEqual(tau * F._HALF_DIVISOR, exp, places=4)

    def test_halflife_definition(self) -> None:
        for tau in F.TAUS:
            self.assertAlmostEqual(F.retention_component(tau * F._HALF_DIVISOR,
                                                          tau), 0.5, places=6)

    def test_monotone_decay(self) -> None:
        for tau in F.TAUS:
            vals = [F.retention_component(t, tau) for t in (0, 1, 7, 30, 365)]
            for a, b in zip(vals, vals[1:]):
                self.assertGreaterEqual(a, b)


class TestComplexityForgetsFirst(unittest.TestCase):
    """复杂度高 ⇒ 任意 age 下保持率都更低（论文的核心机制）。"""

    def test_retention_ordered_by_complexity(self) -> None:
        for age in (1 / 24, 1, 7, 30, 365):
            vals = [F.retention(age, c) for c in (0.0, 0.25, 0.5, 0.75, 1.0)]
            for a, b in zip(vals, vals[1:]):
                self.assertGreater(a, b, f"age={age}")

    def test_halflife_ordered_by_complexity(self) -> None:
        hs = [F.halflife_of(c) for c in (0.0, 0.25, 0.5, 0.75, 1.0)]
        for a, b in zip(hs, hs[1:]):
            self.assertGreater(a, b)

    def test_weights_sum_to_one(self) -> None:
        for c in (0.0, 0.3, 0.7, 1.0):
            self.assertAlmostEqual(sum(F.component_weights(c)), 1.0, places=6)

    def test_profile_threshold(self) -> None:
        pf = ForgettingProfile(threshold=0.3)
        self.assertTrue(pf.forgotten(365, 1.0))   # 高复杂度：1年后已忘
        self.assertFalse(pf.forgotten(365, 0.0))  # 低复杂度：仍在


class TestIntegration(unittest.TestCase):
    def _compressed(self, n: int, forgetting: ForgettingProfile):
        """造一条由 n 条相似输入合并成的 gist。"""
        m = GeneralMemory(min_episodes=2, forgetting=forgetting)
        texts = ["窗前明月光", "床前明月光", "月光明窗前",
                 "明月照窗前", "窗前看明月", "床前看月光"][:n]
        for i, s in enumerate(texts):
            drill(m, s, "疑似地上霜", 2, base=i * 2)
        m.consolidate()
        m.compress()
        return m

    def test_complexity_from_support(self) -> None:
        m = GeneralMemory()
        self.assertEqual(m._complexity(1), 0.0)
        self.assertLess(m._complexity(2), m._complexity(8))
        self.assertLess(m._complexity(8), 1.0)

    def test_high_complexity_forgotten_first(self) -> None:
        """同样放 30 天：高复杂度 gist 被删，低复杂度保留。"""
        pf = ForgettingProfile(threshold=0.5)
        m = GeneralMemory(forgetting=pf, complexity_scale=1.0)
        drill(m, "1+1", "2", 2, 0)          # support=1 → 复杂度 0
        m.consolidate()
        m.compress()                        # 单条不成簇，仍是逐字 → 造第二条
        # 直接构造两条不同复杂度的 gist 更可控
        m.gists = []
        from ultra_instinct.general_memory import GistNode
        import numpy as np
        v = np.array([1.0, 0.0], dtype=np.float32)
        m.gists = [
            GistNode(center=v, response="A", radius=0.5, coherence=1.0,
                     support=1, complexity=0.0, born=0.0),
            GistNode(center=v, response="B", radius=0.5, coherence=1.0,
                     support=20, complexity=1.0, born=0.0),
        ]
        st = m.forget(now=30)
        self.assertEqual(st["forgotten"], 1)
        self.assertEqual(len(m.gists), 1)
        self.assertEqual(m.gists[0].response, "A")
        self.assertEqual(st["enabled"], True)

    def test_gist_weight_decays_with_age(self) -> None:
        m = GeneralMemory(forgetting=ForgettingProfile())
        drill(m, "窗前明月光", "疑似地上霜", 2, 0)
        m.consolidate()
        m.compress()
        fresh = m.query("窗前明月光", now=0).weight
        stale = m.query("窗前明月光", now=365).weight
        self.assertGreater(fresh, stale)

    def test_forget_disabled_by_default(self) -> None:
        m = GeneralMemory()
        st = m.forget(now=100000)
        self.assertEqual(st, {"forgotten": 0, "kept": 0, "enabled": False})

    def test_reconsolidation_after_forget(self) -> None:
        """被忘掉后只能靠新观测重新长出，不是解压。"""
        m = GeneralMemory(min_episodes=2, forgetting=ForgettingProfile(threshold=0.5))
        for i, s in enumerate(("窗前明月光", "床前明月光")):
            drill(m, s, "疑似地上霜", 2, base=i * 2)
        m.consolidate()
        m.compress()
        self.assertEqual(len(m.gists), 1)
        self.assertEqual(m.query("窗前明月光", now=0).source, "gist")

        # 复杂度>0 的 gist 放足够久后被真删
        m.forget(now=3650)
        self.assertEqual(len(m.gists), 0)
        self.assertEqual(m.query("窗前明月光", now=3650).source, "none")

        # 重新观测 2 次 → 逐字长回，不是从要旨解压
        m.reconsolidate("窗前明月光", "疑似地上霜", 100, now=3651)
        m.reconsolidate("窗前明月光", "疑似地上霜", 101, now=3652)
        self.assertEqual(m.query("窗前明月光", now=3653).source, "verbatim")


if __name__ == "__main__":
    unittest.main(verbosity=2)
