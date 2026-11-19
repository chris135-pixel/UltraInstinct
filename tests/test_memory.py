"""记忆表：查询、权重、空洞填充、撤销、持久化。"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from ultra_instinct.memory import LUTMemory
from ultra_instinct.quantize import Grid


def make_memory(size: int = 8, n_skills: int = 1) -> LUTMemory:
    return LUTMemory(grid=Grid(size, 1.0), n_skills=n_skills, act_dim=2)


class TestQuery(unittest.TestCase):
    def test_full_coverage_weight_is_one(self) -> None:
        mem = make_memory()
        for i in range(8):
            for j in range(8):
                mem.set_cell(0, (i, j), np.array([1.0, 0.0]))
        mem.finalize()
        _, w = mem.query(np.array([0.13, -0.42]), 0)
        self.assertAlmostEqual(float(w), 1.0, places=6)

    def test_empty_memory_weight_is_zero(self) -> None:
        mem = make_memory().finalize()
        a, w = mem.query(np.array([0.0, 0.0]), 0)
        self.assertAlmostEqual(float(w), 0.0, places=6)
        np.testing.assert_allclose(a, [0.0, 0.0])

    def test_weight_tracks_coverage(self) -> None:
        """只固化 4 个角时，格中心权重必须是 1，边界处才会降到 0。"""
        mem = make_memory(size=4)
        for i in range(2):
            for j in range(2):
                mem.set_cell(0, (i, j), np.array([0.0, 1.0]))
        mem.finalize()

        _, w_in = mem.query(np.array([-1.0, -1.0]), 0)
        _, w_out = mem.query(np.array([1.0, 1.0]), 0)
        _, w_mid = mem.query(np.array([0.0, 0.0]), 0)

        self.assertAlmostEqual(float(w_in), 1.0, places=6)
        self.assertAlmostEqual(float(w_out), 0.0, places=6)
        self.assertTrue(0.0 < float(w_mid) < 1.0)

    def test_action_from_table(self) -> None:
        mem = make_memory(size=3)
        for i in range(3):
            for j in range(3):
                mem.set_cell(0, (i, j), np.array([float(i), float(j)]))
        mem.finalize()
        a, _ = mem.query(np.array([-1.0, 1.0]), 0)
        np.testing.assert_allclose(a, [0.0, 2.0], atol=1e-6)

    def test_uncovered_corners_excluded(self) -> None:
        """未固化角点不参与加权，其值不会污染结果 —— 这是一次真实 bug 的回归测试。

        第一版先把未固化格填成邻近值再插值，结果外推值混进来：
        朝左上的 delta 输出了朝右上的动作。
        """
        mem = make_memory(size=4)
        mem.set_cell(0, (0, 0), np.array([1.0, 0.0]))
        mem.finalize()

        a_exact, w_exact = mem.query(np.array([-1.0, -1.0]), 0)
        self.assertAlmostEqual(float(w_exact), 1.0, places=5)
        np.testing.assert_allclose(a_exact, [1.0, 0.0], atol=1e-5)

        # 靠近 (1,1)：四个角点里只有 (0,0) 固化，权重很小
        a_mid, w_mid = mem.query(np.array([-0.5, -0.5]), 0)
        self.assertLess(float(w_mid), 0.5)
        np.testing.assert_allclose(a_mid, [1.0, 0.0], atol=1e-5)

    def test_all_corners_uncovered_gives_zero_weight(self) -> None:
        mem = make_memory(size=4)
        mem.set_cell(0, (0, 0), np.array([1.0, 0.0]))
        mem.finalize()
        _, w = mem.query(np.array([1.0, 1.0]), 0)
        self.assertAlmostEqual(float(w), 0.0, places=6)

    def test_batch_and_scalar_agree(self) -> None:
        mem = make_memory(size=6)
        rng = np.random.default_rng(0)
        for i in range(6):
            for j in range(6):
                mem.set_cell(0, (i, j), rng.normal(size=2))
        mem.finalize()

        d = np.array([0.21, -0.33])
        a1, w1 = mem.query(d, 0)
        a2, w2 = mem.query(d[None, :], 0)
        np.testing.assert_allclose(a1, a2[0], atol=1e-6)
        self.assertAlmostEqual(float(w1), float(w2[0]), places=6)

    def test_skills_isolated(self) -> None:
        mem = make_memory(size=4, n_skills=2)
        for i in range(4):
            for j in range(4):
                mem.set_cell(0, (i, j), np.array([1.0, 0.0]))
                mem.set_cell(1, (i, j), np.array([0.0, 1.0]))
        mem.finalize()
        a0, _ = mem.query(np.array([0.0, 0.0]), 0)
        a1, _ = mem.query(np.array([0.0, 0.0]), 1)
        np.testing.assert_allclose(a0, [1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(a1, [0.0, 1.0], atol=1e-6)


class TestInvalidate(unittest.TestCase):
    def test_invalidate_removes_cell(self) -> None:
        mem = make_memory(size=4)
        for i in range(4):
            for j in range(4):
                mem.set_cell(0, (i, j), np.array([1.0, 0.0]))
        mem.finalize()

        idx = mem.invalidate(np.array([-1.0, -1.0]), 0)
        self.assertEqual(idx, (0, 0))
        self.assertLess(mem.coverage[0], 1.0)

    def test_invalidate_twice_returns_none(self) -> None:
        """重复撤销同一格不能重复计数。"""
        mem = make_memory(size=4)
        mem.set_cell(0, (0, 0), np.array([1.0, 0.0]))
        mem.finalize()
        d = np.array([-1.0, -1.0])
        self.assertIsNotNone(mem.invalidate(d, 0))
        self.assertIsNone(mem.invalidate(d, 0))

    def test_invalidate_keeps_other_cells(self) -> None:
        mem = make_memory(size=4)
        for i in range(4):
            for j in range(4):
                mem.set_cell(0, (i, j), np.array([1.0, 0.0]))
        mem.finalize()
        before = mem.coverage[0]
        mem.invalidate(np.array([-1.0, -1.0]), 0)
        self.assertAlmostEqual(float(mem.coverage[0]), float(before) - 1 / 16, places=6)


class TestPersistence(unittest.TestCase):
    def test_roundtrip(self) -> None:
        mem = make_memory(size=5)
        rng = np.random.default_rng(3)
        for i in range(5):
            for j in range(5):
                mem.set_cell(0, (i, j), rng.normal(size=2))
        mem.finalize()

        d = np.array([[0.1, 0.2], [-0.7, 0.4]])
        a0, w0 = mem.query(d, 0)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "mem.npz")
            mem.save(path)
            loaded = LUTMemory.load(path)
            a1, w1 = loaded.query(d, 0)

        np.testing.assert_allclose(a0, a1, atol=1e-6)
        np.testing.assert_allclose(w0, w1, atol=1e-6)
        self.assertEqual(loaded.grid.size, 5)


if __name__ == "__main__":
    unittest.main()
