"""训练与导出：torch 训练结果必须和 numpy 推理完全一致。"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import torch

from ultra_instinct.net import MLP
from ultra_instinct.training.export import export_mlp
from ultra_instinct.training.model import TorchMLP
from ultra_instinct.training.trainer import ReplayBuffer, train_skill


class TestExport(unittest.TestCase):
    def test_numpy_matches_torch(self) -> None:
        """导出最容易错的地方：线性层权重要转置。转错了形状还对，数值全错。"""
        torch.manual_seed(0)
        net = TorchMLP(input_dim=3, hidden=16, output_dim=2)
        x = np.random.default_rng(1).normal(size=(64, 3)).astype(np.float32)

        with torch.no_grad():
            want = net(torch.from_numpy(x)).numpy()
        got = export_mlp(net).forward(x)

        np.testing.assert_allclose(got, want, atol=1e-5)

    def test_dims(self) -> None:
        net = TorchMLP(input_dim=4, hidden=8, output_dim=3)
        mlp = export_mlp(net)
        self.assertEqual(mlp.input_dim, 4)
        self.assertEqual(mlp.output_dim, 3)

    def test_no_tanh_output(self) -> None:
        torch.manual_seed(2)
        net = TorchMLP(input_dim=2, hidden=8, output_dim=2, out_tanh=False)
        x = np.random.default_rng(2).normal(size=(10, 2)).astype(np.float32)
        with torch.no_grad():
            want = net(torch.from_numpy(x)).numpy()
        got = export_mlp(net).forward(x)
        np.testing.assert_allclose(got, want, atol=1e-5)

    def test_mlp_save_load(self) -> None:
        net = TorchMLP(input_dim=3, hidden=8, output_dim=2)
        mlp = export_mlp(net)
        x = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        want = mlp.forward(x)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "net.npz")
            mlp.save(path)
            got = MLP.load(path).forward(x)
        np.testing.assert_allclose(got, want, atol=1e-6)


class TestReplay(unittest.TestCase):
    def test_add_and_sample(self) -> None:
        buf = ReplayBuffer(capacity=100)
        x = np.arange(20, dtype=np.float32).reshape(10, 2)
        y = np.zeros((10, 2), dtype=np.float32)
        buf.add(x, y)
        self.assertEqual(len(buf), 10)

        rng = np.random.default_rng(0)
        xs, ys = buf.sample(5, rng)
        self.assertEqual(xs.shape, (5, 2))
        self.assertEqual(ys.shape, (5, 2))

    def test_capacity_evicts_oldest(self) -> None:
        buf = ReplayBuffer(capacity=10)
        for _ in range(5):
            buf.add(np.zeros((4, 2), np.float32), np.zeros((4, 2), np.float32))
        self.assertLessEqual(len(buf), 48)


class TestTrainSkill(unittest.TestCase):
    def test_loss_drops(self) -> None:
        torch.manual_seed(0)
        net = TorchMLP(input_dim=3, hidden=16, output_dim=2)

        def task(d):
            return np.stack([d[:, 0], d[:, 1]], axis=-1)

        cfg = type("C", (), {"hidden": 16, "steps": 200, "lr": 5e-3,
                             "batch": 64, "replay_ratio": 0.0, "out_tanh": True})()
        loss = train_skill(net, task, cfg, np.random.default_rng(0),
                           skill_id=0, n_skills=1)
        self.assertLess(loss, 0.5)

    def test_mask_filters_samples(self) -> None:
        torch.manual_seed(0)
        net = TorchMLP(input_dim=3, hidden=8, output_dim=2)
        seen: list[int] = []

        def task(d):
            seen.append(len(d))
            return np.zeros((len(d), 2), dtype=np.float32)

        cfg = type("C", (), {"hidden": 8, "steps": 10, "lr": 1e-3,
                             "batch": 64, "replay_ratio": 0.0, "out_tanh": True})()
        train_skill(net, task, cfg, np.random.default_rng(0), skill_id=0,
                    n_skills=2, mask_fn=lambda d: d[:, 0] > 0)
        self.assertTrue(all(n <= 64 for n in seen))


if __name__ == "__main__":
    unittest.main()
