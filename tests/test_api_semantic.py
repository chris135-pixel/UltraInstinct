"""API 语义注入与相似门槛测试：语义命中 + 域外拒答。"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from fastapi.testclient import TestClient

import ultra_instinct.api as api
from ultra_instinct.training.semantic_train import load_pairs, train_semantic, save_weights

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KV = os.path.join(ROOT, "examples", "kv_example.json")


class TestAPISemantic(unittest.TestCase):
    def setUp(self) -> None:
        # 临时训一个语义权重，测试不依赖 checkpoints 里的交付物
        self._tmp = tempfile.TemporaryDirectory()
        self.wp = os.path.join(self._tmp.name, "sem.npz")
        save_weights(train_semantic(load_pairs(KV), steps=200,
                                    hidden=32, out_dim=32, seed=1), self.wp)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _client(self, threshold: float) -> TestClient:
        api.state = api.build_state(semantic_weights=self.wp,
                                    gist_threshold=threshold)
        return TestClient(api.app)

    def test_semantic_encoder_injected(self) -> None:
        s = api.build_state(semantic_weights=self.wp)
        self.assertTrue(s.memory.encoder.trained)
        self.assertEqual(s.memory.encoder.dim, 32)

    def test_high_threshold_rejects_out_of_domain(self) -> None:
        """语义场景 0.85 门槛：同义命中、域外拒答。"""
        c = self._client(threshold=0.85)
        items = [{"content": "它是什么颜色的狗", "response": "它是一条黑色的德牧",
                  "episode_id": k} for k in range(3)]
        r = c.post("/v1/memories", json={"items": items, "consolidate": True})
        self.assertEqual(r.status_code, 200)

        hit = c.post("/v1/chat/completions",
                     json={"messages": [{"role": "user", "content": "那狗是啥色"}],
                           "stream": False}).json()
        self.assertEqual(hit["choices"][0]["message"]["content"],
                         "它是一条黑色的德牧")

        miss = c.post("/v1/chat/completions",
                      json={"messages": [{"role": "user", "content": "今天股票涨了吗"}],
                            "stream": False}).json()
        self.assertEqual(miss["choices"][0]["message"]["content"], "")
        self.assertEqual(miss["usage"]["source"], "none")


if __name__ == "__main__":
    unittest.main()
