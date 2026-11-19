"""网格量化与插值权重。"""

from __future__ import annotations

import unittest

import numpy as np

from ultra_instinct.quantize import Grid


class TestGrid(unittest.TestCase):
    def setUp(self) -> None:
        self.g = Grid(size=5, extent=1.0)

    def test_spacing(self) -> None:
        # 5 个格中心覆盖 [-1, 1]，间距 0.5
        self.assertAlmostEqual(self.g.spacing, 0.5)

    def test_to_coord_endpoints(self) -> None:
        np.testing.assert_allclose(self.g.to_coord(np.array([-1.0, 1.0])), [0.0, 4.0])

    def test_nearest_clamps(self) -> None:
        idx = self.g.nearest(np.array([[-5.0, 5.0], [0.0, 0.0]]))
        np.testing.assert_array_equal(idx, [[0, 4], [2, 2]])

    def test_bilinear_weights_sum_to_one(self) -> None:
        d = np.array([[0.13, -0.42], [0.9, 0.1], [-0.99, 0.99]])
        _, frac, _ = self.g.bilinear(d)
        w = self.g.weights(frac)
        np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-12)

    def test_bilinear_reproduces_linear_field(self) -> None:
        """双线性插值对线性场必须精确 —— 这是它比最近邻准的根本原因。"""
        ax = (np.arange(5) - 2) * 0.5
        gx, gy = np.meshgrid(ax, ax, indexing="ij")
        table = np.stack([2.0 * gx + 3.0 * gy, -gx + 0.5 * gy], axis=-1)

        rng = np.random.default_rng(0)
        d = rng.uniform(-1, 1, (200, 2))
        _, frac, corners = self.g.bilinear(d)
        w = self.g.weights(frac)
        vals = table[corners[:, :, 0], corners[:, :, 1]]
        got = (vals * w[..., None]).sum(axis=1)
        want = np.stack([2.0 * d[:, 0] + 3.0 * d[:, 1],
                         -d[:, 0] + 0.5 * d[:, 1]], axis=-1)
        np.testing.assert_allclose(got, want, atol=1e-12)

    def test_corners_in_range(self) -> None:
        d = np.array([[1.0, 1.0], [-1.0, -1.0], [0.7, -0.3]])
        _, _, corners = self.g.bilinear(d)
        self.assertTrue((corners >= 0).all())
        self.assertTrue((corners <= 4).all())

    def test_invalid_size(self) -> None:
        with self.assertRaises(ValueError):
            Grid(size=1)


if __name__ == "__main__":
    unittest.main()
