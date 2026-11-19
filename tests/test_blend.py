"""动作混合：端点、正交、退化保护。"""

from __future__ import annotations

import unittest

import numpy as np

from ultra_instinct.blend import blend


class TestBlend(unittest.TestCase):
    def test_weight_one_returns_lut(self) -> None:
        lut = np.array([[1.0, 0.0]])
        net = np.array([[0.0, 1.0]])
        out = blend(lut, net, np.array([1.0]))
        np.testing.assert_allclose(out, [[1.0, 0.0]], atol=1e-6)

    def test_weight_zero_returns_net(self) -> None:
        lut = np.array([[1.0, 0.0]])
        net = np.array([[0.0, 1.0]])
        out = blend(lut, net, np.array([0.0]))
        np.testing.assert_allclose(out, [[0.0, 1.0]], atol=1e-6)

    def test_half_weight_on_orthogonal(self) -> None:
        """正交两路各占一半，归一化后应是 45 度方向。"""
        lut = np.array([[1.0, 0.0]])
        net = np.array([[0.0, 1.0]])
        out = blend(lut, net, np.array([0.5]))
        np.testing.assert_allclose(out, [[1 / np.sqrt(2), 1 / np.sqrt(2)]], atol=1e-6)

    def test_continuity_across_weight(self) -> None:
        """权重扫过整个区间，输出方向必须连续 —— 这是替代硬切换的全部意义。"""
        lut = np.array([[1.0, 0.0]])
        net = np.array([[0.0, 1.0]])
        ws = np.linspace(0.0, 1.0, 201)
        out = blend(np.repeat(lut, len(ws), axis=0),
                    np.repeat(net, len(ws), axis=0), ws)

        cos = (out[:-1] * out[1:]).sum(axis=1)
        ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        self.assertLess(float(ang.max()), 1.0)

    def test_opposite_vectors_no_explosion(self) -> None:
        """两路接近相反时，混合模长趋零，不能归一化放大成随机方向。"""
        lut = np.array([[1.0, 0.0]])
        net = np.array([[-1.0, 0.0]])
        out = blend(lut, net, np.array([0.5]))
        self.assertLess(float(np.linalg.norm(out)), 0.5)

    def test_no_normalize_keeps_magnitude(self) -> None:
        lut = np.array([[2.0, 0.0]])
        net = np.array([[0.0, 2.0]])
        out = blend(lut, net, np.array([0.5]), normalize=False)
        np.testing.assert_allclose(out, [[1.0, 1.0]], atol=1e-6)

    def test_scalar_weight(self) -> None:
        lut = np.array([[1.0, 0.0], [1.0, 0.0]])
        net = np.array([[0.0, 1.0], [0.0, 1.0]])
        out = blend(lut, net, 1.0)
        np.testing.assert_allclose(out, lut, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
