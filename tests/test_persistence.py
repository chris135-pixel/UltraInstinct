"""持久化测试：save/load 往返、损坏文件、API 重启恢复。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import ultra_instinct.api as api
from ultra_instinct import ContentEncoder, GeneralMemory
from ultra_instinct.forgetting import ForgettingProfile
from fastapi.testclient import TestClient


def drill(mem: GeneralMemory, content: str, response: str,
          n: int, base: int) -> None:
    for k in range(n):
        mem.practice(content, response, base + k, now=base + k)


class TestPersistence(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "memory.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_roundtrip_all_tiers(self) -> None:
        """三级记忆 + pending + 缓存全部往返一致。"""
        m = GeneralMemory(ContentEncoder(), min_episodes=2,
                          forgetting=ForgettingProfile())
        drill(m, "1+1", "2", 2, 0)
        drill(m, "2+2", "4", 2, 10)
        m.consolidate()
        drill(m, "3+3", "6", 1, 20)          # 未固化的 pending
        drill(m, "窗前明月光", "疑似地上霜", 2, 30)
        drill(m, "床前明月光", "疑似地上霜", 2, 40)
        m.compress(now=50)                   # 造 gist，删掉对应 verbatim

        m.save(self.path)
        self.assertTrue(self.path.exists())
        r = GeneralMemory.load(self.path)

        self.assertEqual(set(r.verbatim), set(m.verbatim))
        self.assertEqual(len(r.gists), len(m.gists))
        self.assertEqual(len(r.episodic), len(m.episodic))
        self.assertEqual(set(r._pending), set(m._pending))
        self.assertEqual(set(r._feat_cache), set(m._feat_cache))
        self.assertEqual(set(r._text_cache), set(m._text_cache))
        # 来源与恢复前一致即可：压缩可能把 1+1 也卷进 gist 簇，
        # 这是编码粒度现象，不是持久化问题。
        for q in ("1+1", "窗前明月"):
            self.assertEqual(r.query(q).response, m.query(q).response)
            self.assertEqual(r.query(q).source, m.query(q).source)
        # 恢复后能继续巩固 pending
        drill(r, "3+3", "6", 1, 21)
        r.consolidate()
        self.assertEqual(r.query("3+3").response, "6")

    def test_config_restored(self) -> None:
        """构造参数也随文件恢复，不用调用方重新传。"""
        m = GeneralMemory(ContentEncoder(), min_episodes=4,
                          coherence_min=0.75, min_confidence=0.3,
                          forgetting=ForgettingProfile(threshold=0.4,
                                                       min_strength=0.1))
        m.save(self.path)
        r = GeneralMemory.load(self.path)
        self.assertEqual(r.min_episodes, 4)
        self.assertEqual(r.coherence_min, 0.75)
        self.assertEqual(r.min_confidence, 0.3)
        self.assertIsNotNone(r.forgetting)
        self.assertEqual(r.forgetting.threshold, 0.4)
        self.assertEqual(r.forgetting.min_strength, 0.1)

    def test_no_forgetting_restored_as_none(self) -> None:
        m = GeneralMemory(ContentEncoder(), forgetting=None)
        m.save(self.path)
        r = GeneralMemory.load(self.path)
        self.assertIsNone(r.forgetting)

    def test_atomic_no_leftover_tmp(self) -> None:
        """正常保存后目录里只有目标文件，没有临时文件残留。"""
        m = GeneralMemory(ContentEncoder())
        drill(m, "1+1", "2", 2, 0)
        m.consolidate()
        m.save(self.path)
        leftovers = [p.name for p in self.path.parent.iterdir()
                     if p.name.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_corrupt_file_raises(self) -> None:
        """损坏文件要明确报错，不静默返回空记忆。"""
        self.path.write_text("{ broken json", encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError):
            GeneralMemory.load(self.path)

    def test_bad_version_raises(self) -> None:
        self.path.write_text(json.dumps({"version": 99}), encoding="utf-8")
        with self.assertRaises(ValueError):
            GeneralMemory.load(self.path)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            GeneralMemory.load(self.path)

    def test_api_restart_restores_memory(self) -> None:
        """模拟服务重启：新 State 从文件恢复，能查到之前练的内容。"""
        old = api.build_state(memory_file=self.path)
        old_state = api.state
        try:
            api.state = old
            client = TestClient(api.app)
            items = [{"content": "1+1", "response": "2",
                      "episode_id": i, "now": float(i)} for i in range(3)]
            resp = client.post("/v1/memories", json={"items": items})
            self.assertTrue(resp.json()["saved"])
            self.assertTrue(self.path.exists())

            # 重启：全新 state 从文件加载
            api.state = api.build_state(memory_file=self.path,
                                        load_if_exists=True)
            client2 = TestClient(api.app)
            body = client2.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "1+1"}],
            }).json()
            self.assertEqual(body["choices"][0]["message"]["content"], "2")
            self.assertEqual(body["usage"]["source"], "verbatim")
            stats = client2.get("/v1/memories/stats").json()
            self.assertEqual(stats["persist_file"], str(self.path))
        finally:
            api.state = old_state

    def test_api_no_persist_stays_in_memory(self) -> None:
        """未配置路径时 saved=False，不碰磁盘。"""
        old_state = api.state
        try:
            api.state = api.build_state()   # 无 memory_file
            client = TestClient(api.app)
            items = [{"content": "1+1", "response": "2",
                      "episode_id": i, "now": 0.0} for i in range(3)]
            resp = client.post("/v1/memories", json={"items": items}).json()
            self.assertFalse(resp["saved"])
            self.assertEqual(api.state.last_saved, None)
        finally:
            api.state = old_state

    def test_compress_then_save_restores_gist(self) -> None:
        """压缩后落盘，重启后 gist 路径仍可用。"""
        old_state = api.state
        try:
            api.state = api.build_state(memory_file=self.path)
            api.state.memory.gradual_decay = False  # 锚定旧语义：练完立即压缩造 gist
            client = TestClient(api.app)
            items = []
            ep = 0
            for content in ("窗前明月光", "床前明月光", "月光明窗前"):
                for _ in range(3):
                    items.append({"content": content, "response": "疑似地上霜",
                                  "episode_id": ep, "now": float(ep)})
                    ep += 1
            client.post("/v1/memories", json={"items": items})
            client.post("/v1/memories", json={"items": [], "compress": True})

            api.state = api.build_state(memory_file=self.path,
                                        load_if_exists=True)
            client2 = TestClient(api.app)
            stats = client2.get("/v1/memories/stats").json()
            self.assertGreater(stats["gists"], 0)
            body = client2.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "窗前明月光"}],
            }).json()
            self.assertEqual(body["choices"][0]["message"]["content"],
                             "疑似地上霜")
        finally:
            api.state = old_state


if __name__ == "__main__":
    unittest.main(verbosity=2)
