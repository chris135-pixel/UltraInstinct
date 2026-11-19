"""语义编码器测试：可注入、退化行为、训练后同义/异义分离、持久化。"""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from ultra_instinct import GeneralMemory
from ultra_instinct.encoding import ContentEncoder
from ultra_instinct.net import MLP
from ultra_instinct.semantic import SemanticEncoder
from ultra_instinct.training.semantic_train import (
    load_pairs, save_weights, supcon_loss, train_semantic,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KV = os.path.join(ROOT, "examples", "kv_example.json")


def quick_encoder(seed: int = 0) -> SemanticEncoder:
    """训练一个小网络，供多个测试共用（各测试自建，互不影响）。"""
    weights = train_semantic(load_pairs(KV), steps=250,
                             hidden=32, out_dim=32, seed=seed)
    fd, path = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    save_weights(weights, path)
    enc = SemanticEncoder.from_weights(path)
    os.unlink(path)
    return enc


class TestSemanticEncoder(unittest.TestCase):
    def test_untrained_degenerates_to_char_features(self) -> None:
        """未训练时与 ContentEncoder 逐位一致——保证不训练也能用。"""
        enc = SemanticEncoder()
        base = ContentEncoder()
        self.assertFalse(enc.trained)
        for t in ("它是什么狗", "那狗是啥色", "1+1", ""):
            np.testing.assert_allclose(enc.features(t), base.features(t))
            self.assertEqual(enc.key_of(t), base.key_of(t))
            self.assertEqual(enc.clean(t), base.clean(t))

    def test_trained_similarity_separates_synonyms(self) -> None:
        """训练后：同义问法显著拉近，异义被推远。"""
        before = SemanticEncoder()
        after = quick_encoder()
        s_syn_b = before.similarity(before.features("它是什么狗"),
                                    before.features("那狗是啥色"))
        s_syn_a = after.similarity(after.features("它是什么狗"),
                                   after.features("那狗是啥色"))
        s_ant_a = after.similarity(after.features("它是什么颜色的狗"),
                                   after.features("1+1"))
        self.assertGreater(s_syn_a, s_syn_b + 0.3, "同义组应被显著拉近")
        self.assertGreater(s_syn_a, 0.8, "同义组应高度相似")
        self.assertLess(s_ant_a, s_syn_a - 0.5, "异义组应被推远")

    def test_dimension_mismatch_rejected(self) -> None:
        """网络输入维度与 base 不符时必须报错，而不是静默算错。"""
        w = train_semantic(load_pairs(KV), steps=5, hidden=8, out_dim=8)
        mlp = MLP(w1=w["w1"], b1=w["b1"], w2=w["w2"], b2=w["b2"],
                  w3=w["w3"], b3=w["b3"], out_tanh=False)
        with self.assertRaises(ValueError):
            SemanticEncoder(base=ContentEncoder(dim=64), mlp=mlp)

    def test_injectable_and_synonym_hit(self) -> None:
        """语义编码器可直接注入 GeneralMemory；没练过的同义问法能命中。"""
        enc = quick_encoder()
        mem = GeneralMemory(enc, min_episodes=2, gist_threshold=0.4)
        for k in range(2):
            mem.practice("它是什么颜色的狗", "它是一条黑色的德牧", k, now=k)
        mem.consolidate()
        exact = mem.query("它是什么颜色的狗")
        self.assertEqual(exact.source, "verbatim")
        self.assertEqual(exact.response, "它是一条黑色的德牧")
        syn = mem.query("那狗是啥色")
        self.assertEqual(syn.response, "它是一条黑色的德牧")
        self.assertIn(syn.source, ("episodic", "gist"))

    def test_untrained_injection_still_works(self) -> None:
        """不训练也能注入并正常逐字命中（退化为字符特征）。"""
        mem = GeneralMemory(SemanticEncoder(), min_episodes=1)
        mem.practice("1+1", "2", 0, now=0)
        mem.consolidate()
        self.assertEqual(mem.query("1+1").response, "2")

    def test_save_load_with_injected_encoder(self) -> None:
        """save 后显式注入同一编码器 load，逐字命中保持。"""
        enc = quick_encoder()
        mem = GeneralMemory(enc, min_episodes=2)
        for k in range(2):
            mem.practice("它是什么颜色的狗", "它是一条黑色的德牧", k, now=k)
        mem.consolidate()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.json")
            mem.save(p)
            restored = GeneralMemory.load(p, encoder=enc)
            self.assertTrue(restored.encoder.trained)
            self.assertEqual(restored.query("它是什么颜色的狗").response,
                             "它是一条黑色的德牧")

    def test_weights_roundtrip(self) -> None:
        """权重 npz 往返后特征逐位一致。"""
        enc = quick_encoder()
        a = enc.features("它是什么狗")
        self.assertTrue(enc.trained)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "w.npz")
            enc.save_weights(p)
            enc2 = SemanticEncoder.from_weights(p)
            np.testing.assert_allclose(enc2.features("它是什么狗"), a)

    def test_no_weights_to_save(self) -> None:
        """未训练时存权重应报错，避免产出空文件。"""
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                SemanticEncoder().save_weights(os.path.join(d, "w.npz"))

    def test_supcon_needs_positive_pairs(self) -> None:
        """一个答案只有一条样本时无正样本对，训练应明确报错。"""
        with self.assertRaises(ValueError):
            train_semantic({"A": [], "B": []}, steps=2)


if __name__ == "__main__":
    unittest.main()
