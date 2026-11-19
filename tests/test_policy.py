"""策略路由与 episode 级自检。"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from ultra_instinct.memory import LUTMemory
from ultra_instinct.net import MLP
from ultra_instinct.policy import Policy, PolicyConfig
from ultra_instinct.quantize import Grid


def make_net() -> MLP:
    """3 输入 2 输出的固定网络，权重小，输出近似线性。"""
    rng = np.random.default_rng(0)
    return MLP(
        w1=rng.normal(0, 0.3, (3, 8)).astype(np.float32), b1=np.zeros(8, np.float32),
        w2=rng.normal(0, 0.3, (8, 8)).astype(np.float32), b2=np.zeros(8, np.float32),
        w3=rng.normal(0, 0.3, (8, 2)).astype(np.float32), b3=np.zeros(2, np.float32),
        out_tanh=True,
    )


def make_memory(size: int = 4, fill: bool = True) -> LUTMemory:
    mem = LUTMemory(grid=Grid(size, 1.0), n_skills=1, act_dim=2)
    if fill:
        for i in range(size):
            for j in range(size):
                mem.set_cell(0, (i, j), np.array([1.0, 0.0]))
    return mem.finalize()


class TestRouting(unittest.TestCase):
    def test_no_memory_uses_net(self) -> None:
        p = Policy(net=make_net(), memory=None, config=PolicyConfig())
        out = p.act(np.array([0.3, -0.2]))
        self.assertEqual(out.source, "net")
        self.assertTrue(out.net_called)

    def test_full_coverage_skips_net(self) -> None:
        p = Policy(net=make_net(), memory=make_memory(), config=PolicyConfig())
        out = p.act(np.array([0.3, -0.2]))
        self.assertEqual(out.source, "lut")
        self.assertFalse(out.net_called)
        np.testing.assert_allclose(out.action, [1.0, 0.0], atol=1e-5)

    def test_partial_coverage_blends(self) -> None:
        mem = LUTMemory(grid=Grid(4, 1.0), n_skills=1, act_dim=2)
        for i in range(2):
            for j in range(2):
                mem.set_cell(0, (i, j), np.array([1.0, 0.0]))
        mem.finalize()

        p = Policy(net=make_net(), memory=mem, config=PolicyConfig())
        out = p.act(np.array([0.0, 0.0]))     # 四角里一对一空
        self.assertEqual(out.source, "blend")
        self.assertTrue(out.net_called)
        self.assertTrue(0.0 < out.weight < 1.0)

    def test_empty_region_uses_net(self) -> None:
        mem = LUTMemory(grid=Grid(4, 1.0), n_skills=1, act_dim=2)
        mem.set_cell(0, (0, 0), np.array([1.0, 0.0]))
        mem.finalize()

        p = Policy(net=make_net(), memory=mem, config=PolicyConfig())
        out = p.act(np.array([1.0, 1.0]))
        self.assertEqual(out.source, "net")
        self.assertAlmostEqual(out.weight, 0.0, places=5)

    def test_memory_only_inertia(self) -> None:
        mem = LUTMemory(grid=Grid(4, 1.0), n_skills=1, act_dim=2)
        for i in range(2):
            for j in range(2):
                mem.set_cell(0, (i, j), np.array([0.0, 1.0]))
        mem.finalize()

        p = Policy(net=None, memory=mem, config=PolicyConfig())
        first = p.act(np.array([-1.0, -1.0]))
        self.assertEqual(first.source, "lut")

        out = p.act(np.array([1.0, 1.0]))
        self.assertEqual(out.source, "inertia")
        np.testing.assert_allclose(out.action, first.action, atol=1e-5)

    def test_force_net(self) -> None:
        p = Policy(net=make_net(), memory=make_memory(), config=PolicyConfig())
        p.force_net(0, True)
        out = p.act(np.array([0.3, -0.2]))
        self.assertEqual(out.source, "fallback")
        self.assertTrue(out.net_called)


class TestRateLimit(unittest.TestCase):
    def test_sharp_turn_is_clamped(self) -> None:
        mem = LUTMemory(grid=Grid(2, 1.0), n_skills=1, act_dim=2)
        mem.set_cell(0, (0, 0), np.array([1.0, 0.0]))
        mem.set_cell(0, (1, 1), np.array([0.0, 1.0]))
        mem.set_cell(0, (0, 1), np.array([1.0, 0.0]))
        mem.set_cell(0, (1, 0), np.array([0.0, 1.0]))
        mem.finalize()

        p = Policy(net=None, memory=mem, config=PolicyConfig(max_turn_deg=8.0))
        p.act(np.array([-1.0, -1.0]))        # 朝 x
        out = p.act(np.array([1.0, 1.0]))    # 朝 y，一次转 90 度
        self.assertTrue(out.clamped)


class TestEpisodeCheck(unittest.TestCase):
    def test_success_keeps_memory(self) -> None:
        mem = make_memory()
        p = Policy(net=make_net(), memory=mem, config=PolicyConfig())
        p.begin_episode(0)
        p.act(np.array([-1.0, -1.0]))
        before = float(mem.coverage[0])
        n = p.end_episode(True, 0)
        self.assertEqual(n, 0)
        self.assertAlmostEqual(float(mem.coverage[0]), before, places=6)

    def test_failure_invalidates_hit_cells(self) -> None:
        mem = make_memory()
        p = Policy(net=make_net(), memory=mem, config=PolicyConfig())
        p.begin_episode(0)
        p.act(np.array([-1.0, -1.0]))     # 命中 (0,0)
        p.act(np.array([-0.5, -1.0]))     # 命中 (0,0) 附近
        before = float(mem.coverage[0])
        n = p.end_episode(False, 0)
        self.assertGreater(n, 0)
        self.assertLess(float(mem.coverage[0]), before)
        self.assertEqual(p.invalidated_count[0], n)

    def test_failure_only_touches_lut_cells(self) -> None:
        """走网络算出来的决策不该被撤销 —— 它本来就没进表。"""
        mem = LUTMemory(grid=Grid(4, 1.0), n_skills=1, act_dim=2)
        mem.set_cell(0, (0, 0), np.array([1.0, 0.0]))
        mem.finalize()

        p = Policy(net=make_net(), memory=mem, config=PolicyConfig())
        p.begin_episode(0)
        p.act(np.array([1.0, 1.0]))       # 未覆盖区域，走网络
        n = p.end_episode(False, 0)
        self.assertEqual(n, 0)

    def test_hits_cleared_between_episodes(self) -> None:
        mem = make_memory()
        p = Policy(net=make_net(), memory=mem, config=PolicyConfig())
        p.begin_episode(0)
        p.act(np.array([-1.0, -1.0]))
        p.end_episode(True, 0)

        p.begin_episode(0)
        n = p.end_episode(False, 0)       # 这一轮没做过决策
        self.assertEqual(n, 0)


class TestPersistence(unittest.TestCase):
    def test_roundtrip(self) -> None:
        p = Policy(net=make_net(), memory=make_memory(), config=PolicyConfig(grid_size=4))
        d = np.array([0.2, -0.6])
        a0 = p.act(d).action

        with tempfile.TemporaryDirectory() as td:
            p.save(td)
            loaded = Policy.load(td)
            a1 = loaded.act(d).action

        np.testing.assert_allclose(a0, a1, atol=1e-6)
        self.assertEqual(loaded.config.grid_size, 4)
        self.assertIsNotNone(loaded.memory)

    def test_load_net_only(self) -> None:
        p = Policy(net=make_net(), memory=None, config=PolicyConfig())
        with tempfile.TemporaryDirectory() as td:
            p.save(td)
            loaded = Policy.load(td)
            self.assertIsNotNone(loaded.net)
            self.assertIsNone(loaded.memory)


if __name__ == "__main__":
    unittest.main()
