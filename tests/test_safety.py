"""安全层：限幅、球面插值、进展判据。"""

from __future__ import annotations

import unittest

import numpy as np

from ultra_instinct.safety import ProgressMonitor, RateLimiter, angle_deg, slerp


class TestAngle(unittest.TestCase):
    def test_right_angle(self) -> None:
        self.assertAlmostEqual(angle_deg(np.array([1.0, 0.0]),
                                         np.array([0.0, 1.0])), 90.0, places=6)

    def test_same_direction(self) -> None:
        self.assertAlmostEqual(angle_deg(np.array([1.0, 0.0]),
                                         np.array([3.0, 0.0])), 0.0, places=6)

    def test_opposite(self) -> None:
        self.assertAlmostEqual(angle_deg(np.array([1.0, 0.0]),
                                         np.array([-1.0, 0.0])), 180.0, places=6)


class TestSlerp(unittest.TestCase):
    def test_endpoints(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        np.testing.assert_allclose(slerp(a, b, 0.0), a, atol=1e-6)
        np.testing.assert_allclose(slerp(a, b, 1.0), b, atol=1e-6)

    def test_midpoint_angle(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        m = slerp(a, b, 0.5)
        self.assertAlmostEqual(angle_deg(a, m), 45.0, places=5)

    def test_keeps_unit_length(self) -> None:
        a = np.array([2.0, 0.0])
        b = np.array([0.0, 0.5])
        self.assertAlmostEqual(float(np.linalg.norm(slerp(a, b, 0.3))), 1.0, places=6)


class TestRateLimiter(unittest.TestCase):
    def test_first_action_passes(self) -> None:
        lim = RateLimiter(max_turn_deg=8.0)
        out, clamped = lim.apply(np.array([1.0, 0.0]))
        self.assertFalse(clamped)
        np.testing.assert_allclose(out, [1.0, 0.0], atol=1e-6)

    def test_small_turn_passes(self) -> None:
        lim = RateLimiter(max_turn_deg=10.0)
        lim.apply(np.array([1.0, 0.0]))
        out, clamped = lim.apply(np.array([np.cos(np.radians(5)), np.sin(np.radians(5))]))
        self.assertFalse(clamped)
        self.assertAlmostEqual(angle_deg(np.array([1.0, 0.0]), out), 5.0, places=4)

    def test_large_turn_is_clamped(self) -> None:
        lim = RateLimiter(max_turn_deg=8.0)
        lim.apply(np.array([1.0, 0.0]))
        out, clamped = lim.apply(np.array([0.0, 1.0]))
        self.assertTrue(clamped)
        self.assertAlmostEqual(angle_deg(np.array([1.0, 0.0]), out), 8.0, places=4)

    def test_successive_clamps_accumulate(self) -> None:
        """连续给相反方向，每步最多转 8 度，不能一步跳过去。"""
        lim = RateLimiter(max_turn_deg=8.0)
        prev = lim.apply(np.array([1.0, 0.0]))[0]
        for _ in range(5):
            cur, _ = lim.apply(np.array([0.0, 1.0]))
            self.assertLessEqual(angle_deg(prev, cur), 8.0 + 1e-4)
            prev = cur

    def test_reset_clears_state(self) -> None:
        lim = RateLimiter(max_turn_deg=8.0)
        lim.apply(np.array([1.0, 0.0]))
        lim.reset()
        _, clamped = lim.apply(np.array([0.0, 1.0]))
        self.assertFalse(clamped)

    def test_zero_disables_clamping(self) -> None:
        lim = RateLimiter(max_turn_deg=0.0)
        lim.apply(np.array([1.0, 0.0]))
        _, clamped = lim.apply(np.array([0.0, 1.0]))
        self.assertFalse(clamped)


class TestProgressMonitor(unittest.TestCase):
    def test_reaching_keeps_quiet(self) -> None:
        mon = ProgressMonitor(patience=8)
        for d in np.linspace(1.0, 0.01, 40):
            self.assertFalse(mon.observe(d))

    def test_plateau_triggers(self) -> None:
        mon = ProgressMonitor(patience=8)
        for _ in range(3):
            mon.observe(0.5)
        fired = False
        for _ in range(10):
            fired = mon.observe(0.5) or fired
        self.assertTrue(fired)

    def test_short_plateau_does_not_trigger(self) -> None:
        """绕行时的短暂平台不能误判 —— 这是第一版实现的致命错误。"""
        mon = ProgressMonitor(patience=8)
        mon.observe(0.5)
        for _ in range(5):
            self.assertFalse(mon.observe(0.5))
        for d in np.linspace(0.49, 0.2, 10):
            self.assertFalse(mon.observe(d))

    def test_new_low_resets_counter(self) -> None:
        mon = ProgressMonitor(patience=3)
        mon.observe(0.5)
        mon.observe(0.5)
        mon.observe(0.5)
        mon.observe(0.4)          # 创新低，计数清零
        self.assertFalse(mon.observe(0.4))
        self.assertFalse(mon.observe(0.4))


if __name__ == "__main__":
    unittest.main()
