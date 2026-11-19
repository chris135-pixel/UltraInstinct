"""分级记忆的单元测试。

重点覆盖四件容易悄悄错掉的事：

1. 仿射拟合必须精确还原线性场（否则「保留主旨」是空话）
2. coherence 门槛必须拒绝矛盾块（否则压缩会把两条通路平均成废方向）
3. 要旨半径必须铺满，不能在块与块之间留缝
   （第一版按支撑点范围定半径，b=2 时误差 72°、p95 180°，全是空响）
4. 查询优先级：逐字 > 要旨 > 浅记忆
"""

from __future__ import annotations

import unittest

import numpy as np

from ultra_instinct.hierarchy import (
    EpisodicBuffer,
    GistNode,
    Hierarchy,
    compress,
    query_gists,
)
from ultra_instinct.memory import LUTMemory
from ultra_instinct.quantize import Grid


def make_mem(size: int = 8, fill=None) -> LUTMemory:
    mem = LUTMemory(Grid(size, 1.0), 1, 2)
    for i in range(size):
        for j in range(size):
            x = np.array([i, j], dtype=np.float64) / (size - 1) * 2 - 1
            mem.set_cell(0, (i, j), fill(x) if fill else np.array([1.0, 0.0]))
    return mem


class TestCompress(unittest.TestCase):
    def test_affine_field_recovered_exactly(self) -> None:
        """线性动作场必须被仿射要旨零误差还原。"""
        jac = np.array([[0.5, -0.3], [0.2, 0.8]])
        bias = np.array([0.1, -0.2])
        mem = make_mem(8, fill=lambda x: jac @ x + bias)
        nodes, stats = compress(mem, block=2, min_support=4, coherence_min=0.6)
        self.assertGreater(len(nodes), 0)
        self.assertEqual(stats["rejected"], 0)
        for n in nodes:
            self.assertGreater(n.coherence, 0.999)
            self.assertLess(n.curvature, 1e-4)
            probe = n.center + np.array([0.02, -0.03])
            np.testing.assert_allclose(n.evaluate(probe)[0], jac @ probe + bias,
                                       atol=1e-4)

    def test_coherence_rejects_contradictory_block(self) -> None:
        """块内动作相反时必须拒绝合并，逐字格原样保留。

        用 (i+j) 奇偶决定正负，保证**每一个** 2×2 块都是 2 正 2 负。
        第一版用 y>=0 分界，结果分界线正好落在块边界上，一个矛盾块都没有 ——
        这种对齐巧合在真实系统里不会发生，但会让测试假通过。
        """
        mem = make_mem(8)
        for i in range(8):
            for j in range(8):
                if i < 4:
                    a = np.array([1.0, 0.0])
                else:
                    a = (np.array([1.0, 0.0]) if (i + j) % 2 == 0
                         else np.array([-1.0, 0.0]))
                mem.set_cell(0, (i, j), a)
        before = int((mem.occupancy > 0.5).sum())
        nodes, stats = compress(mem, block=2, min_support=4, coherence_min=0.6)
        after = int((mem.occupancy > 0.5).sum())
        self.assertEqual(stats["rejected"], 8, "右半 8 个块全部矛盾，必须拒绝")
        self.assertEqual(len(nodes), 8, "左半 8 个块一致，应全部合并")
        self.assertLess(after, before)
        for n in nodes:
            self.assertGreaterEqual(n.coherence, 0.6)

    def test_constant_block_is_compressed(self) -> None:
        """块内完全一致时应当合并，且 coherence=1。"""
        mem = make_mem(8)
        nodes, stats = compress(mem, block=2, min_support=4, coherence_min=0.6)
        self.assertEqual(stats["rejected"], 0)
        self.assertTrue(all(n.coherence == 1.0 for n in nodes))
        self.assertTrue(all(n.curvature < 1e-9 for n in nodes))

    def test_radius_tiles_without_gaps(self) -> None:
        """相邻要旨之间不能留缝，否则查询会返回空。"""
        mem = make_mem(8)
        nodes, _ = compress(mem, block=2, min_support=4, coherence_min=0.6)
        centers = sorted([n.center for n in nodes], key=lambda c: (c[0], c[1]))
        checked = 0
        for a in centers:
            for b in centers:
                if a is b:
                    continue
                mid = (a + b) / 2.0
                if np.linalg.norm(mid - a) > 1e-9:
                    act, w = query_gists(nodes, mid, 0)
                    if act is not None:
                        checked += 1
        self.assertGreater(checked, 0)
        # 网格中点必须被覆盖
        for mid in (np.array([-0.57, 0.0]), np.array([0.0, 0.57])):
            act, _ = query_gists(nodes, mid, 0)
            self.assertIsNotNone(act, f"中点 {mid} 没有被任何要旨覆盖")

    def test_footprint_shrinks(self) -> None:
        mem = make_mem(16, fill=lambda x: np.array([x[0], x[1]]))
        before = int((mem.occupancy > 0.5).sum()) * 4
        nodes, _ = compress(mem, block=4, min_support=4, coherence_min=0.6)
        after = int((mem.occupancy > 0.5).sum()) * 4 + sum(n.n_params for n in nodes)
        self.assertLess(after, before * 0.3)


class TestEpisodic(unittest.TestCase):
    def test_power_law_decay(self) -> None:
        buf = EpisodicBuffer(capacity=8, tau=3.0, alpha=1.0)
        buf.write(np.zeros(2), np.array([1.0, 0.0]), 0, 0, now=0.0)
        t = buf.traces[0]
        self.assertAlmostEqual(buf.strength(t, 0.0), 1.0)
        self.assertAlmostEqual(buf.strength(t, 3.0), 0.5)
        self.assertAlmostEqual(buf.strength(t, 9.0), 0.25)
        self.assertGreater(buf.strength(t, 1.0), buf.strength(t, 2.0))

    def test_capacity_evicts_weakest(self) -> None:
        buf = EpisodicBuffer(capacity=2, tau=3.0, alpha=1.0)
        buf.write(np.array([0.0, 0.0]), np.array([1.0, 0.0]), 0, 0, now=0.0)
        buf.write(np.array([0.5, 0.0]), np.array([1.0, 0.0]), 0, 1, now=0.1)
        buf.write(np.array([1.0, 0.0]), np.array([1.0, 0.0]), 0, 2, now=0.2)
        self.assertEqual(len(buf), 2)
        kept = [tuple(np.round(t.delta, 3)) for t in buf.traces]
        self.assertNotIn((0.0, 0.0), kept, "最弱（最旧）的应被淘汰")


class TestHierarchy(unittest.TestCase):
    def _h(self) -> Hierarchy:
        return Hierarchy(Grid(8, 1.0), 1, 2, min_episodes=3)

    def test_consolidate_needs_cross_episode(self) -> None:
        h = self._h()
        d = np.array([0.3, 0.3])
        for ep in range(2):
            h.practice(d, np.array([1.0, 0.0]), 0, ep, now=ep)
        self.assertEqual(h.consolidate(), 0)
        h.practice(d, np.array([1.0, 0.0]), 0, 2, now=2)
        self.assertEqual(h.consolidate(), 1)

    def test_same_episode_repeats_do_not_count(self) -> None:
        h = self._h()
        d = np.array([0.3, 0.3])
        for _ in range(10):
            h.practice(d, np.array([1.0, 0.0]), 0, 0, now=0)
        self.assertEqual(h.consolidate(), 0)

    def test_query_priority_verbatim_first(self) -> None:
        h = self._h()
        h.verbatim.set_cell(0, (5, 5), np.array([1.0, 0.0]))
        h.gists = [GistNode(skill_id=0, center=np.array([5 / 7, 5 / 7]) * 2 - 1,
                            action0=np.array([-1.0, 0.0]), jac=np.zeros((2, 2)),
                            radius=1.0, coherence=1.0, curvature=0.0, support=9)]
        act, w, src = h.query(np.array([5 / 7, 5 / 7]) * 2 - 1, 0)
        self.assertEqual(src, "verbatim")
        np.testing.assert_allclose(act, [1.0, 0.0], atol=1e-6)

    def test_gist_used_when_verbatim_absent(self) -> None:
        h = self._h()
        c = np.array([0.2, 0.2])
        h.gists = [GistNode(skill_id=0, center=c, action0=np.array([0.0, 1.0]),
                            jac=np.zeros((2, 2)), radius=0.3, coherence=0.9,
                            curvature=0.01, support=9)]
        act, w, src = h.query(c, 0)
        self.assertEqual(src, "gist")
        np.testing.assert_allclose(act, [0.0, 1.0], atol=1e-6)
        self.assertLess(w, 1.0, "要旨的信任度不应达到逐字水平")

    def test_expand_regrows_verbatim(self) -> None:
        h = self._h()
        d = np.array([0.4, -0.2])
        idx = tuple(int(v) for v in h.grid.nearest(d))
        self.assertFalse(h.expand(d, 0, np.array([1.0, 0.0]), 0))
        h.expand(d, 0, np.array([1.0, 0.0]), 1)
        self.assertTrue(h.expand(d, 0, np.array([1.0, 0.0]), 2))
        self.assertEqual(h.verbatim.occupancy[(0,) + idx], 1.0)

    def test_invalidate_verbatim(self) -> None:
        h = self._h()
        d = np.array([0.4, -0.2])
        h.verbatim.set_cell(0, tuple(int(v) for v in h.grid.nearest(d)),
                            np.array([1.0, 0.0]))
        self.assertTrue(h.invalidate(d, 0))
        self.assertFalse(h.invalidate(d, 0), "已经撤过的格子不应重复计数")

    def test_invalidate_discounts_gist(self) -> None:
        h = self._h()
        c = np.array([0.1, 0.1])
        node = GistNode(skill_id=0, center=c, action0=np.array([1.0, 0.0]),
                        jac=np.zeros((2, 2)), radius=0.3, coherence=0.9,
                        curvature=0.0, support=9)
        h.gists = [node]
        self.assertTrue(h.invalidate(c, 0))
        self.assertAlmostEqual(node.coherence, 0.45)

    def test_footprint_counts_three_levels(self) -> None:
        h = self._h()
        h.episodic.write(np.zeros(2), np.array([1.0, 0.0]), 0, 0, now=0)
        h.verbatim.set_cell(0, (3, 3), np.array([1.0, 0.0]))
        fp = h.footprint
        self.assertGreater(fp["episodic"], 0)
        self.assertGreater(fp["verbatim"], 0)
        self.assertEqual(fp["total"], fp["verbatim"] + fp["gist"] + fp["episodic"])


if __name__ == "__main__":
    unittest.main()
