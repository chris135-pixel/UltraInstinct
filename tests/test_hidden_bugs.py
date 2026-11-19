"""隐性 bug 回归测试：持久化 encoder 配置、reconsolidate 语义、
uvicorn 直启持久化、查询统计落盘、缓存清理。每个用例对应一个已修复的缺陷。
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import ultra_instinct.api as api
from fastapi.testclient import TestClient

from ultra_instinct import ContentEncoder, GeneralMemory


class TestEncoderPersistence(unittest.TestCase):
    """Bug 2: save/load 必须带上 encoder 配置，否则维度错配或静默查不到。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "m.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_custom_dim_roundtrip_no_crash(self) -> None:
        m = GeneralMemory(ContentEncoder(dim=64), min_episodes=1,
                          gist_threshold=0.1)
        for c in ("窗前明月光", "床前明月光"):
            m.practice(c, "疑似地上霜", len(m.episodic))
        m.consolidate()
        m.compress()
        m.save(self.path)

        r = GeneralMemory.load(self.path)
        self.assertEqual(r.encoder.dim, 64)
        # gist / episodic 路径都不能崩，且能答出
        for q in ("窗前明月光", "窗前明月"):
            ans = r.query(q)
            self.assertEqual(ans.response, "疑似地上霜")

    def test_normalize_flag_roundtrip(self) -> None:
        m = GeneralMemory(ContentEncoder(normalize=False), min_episodes=1)
        m.practice("ＡＢＣ", "答案X", 0)
        m.consolidate()
        m.save(self.path)
        r = GeneralMemory.load(self.path)
        self.assertFalse(r.encoder.normalize)
        self.assertEqual(r.query("ＡＢＣ").response, "答案X")

    def test_ngrams_roundtrip(self) -> None:
        m = GeneralMemory(ContentEncoder(ngrams=(1, 2)), min_episodes=1)
        m.practice("窗前明月光", "疑似地上霜", 0)
        m.consolidate()
        m.save(self.path)
        r = GeneralMemory.load(self.path)
        self.assertEqual(r.encoder.ngrams, (1, 2))

    def test_legacy_file_without_encoder(self) -> None:
        m = GeneralMemory(ContentEncoder(), min_episodes=1)
        m.practice("旧数据", "旧答案", 0)
        m.consolidate()
        m.save(self.path)
        doc = json.loads(self.path.read_text(encoding="utf-8"))
        doc["config"].pop("encoder", None)
        self.path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        r = GeneralMemory.load(self.path)
        self.assertEqual(r.query("旧数据").response, "旧答案")


class TestReconsolidate(unittest.TestCase):
    """Bug 3: 只固化当前 key，返回真实结果，不误报、不全局副作用。"""

    def test_no_global_side_effect(self) -> None:
        m = GeneralMemory(min_episodes=1)
        m.practice("A", "a", 0)
        m.practice("B", "b", 0)          # B 未固化
        m.reconsolidate("A", "a", 1)      # 只应处理 A
        self.assertIn(m.encoder.key_of("A"), m.verbatim)
        self.assertNotIn(m.encoder.key_of("B"), m.verbatim)

    def test_no_false_success(self) -> None:
        m = GeneralMemory(min_episodes=2, stability=0.6)
        for ep, r in ((0, "答案A"), (1, "答案A")):
            m.practice("内容A", r, ep)
        for ep, r in ((0, "X"), (1, "Y"), (2, "Z")):
            m.practice("内容B", r, ep)
        # B 答案不稳定 -> 应返回 False，不能因 A 达标而误报 True
        self.assertFalse(m.reconsolidate("内容B", "W", 3))
        self.assertNotIn(m.encoder.key_of("内容B"), m.verbatim)

    def test_normal_reconsolidation_still_works(self) -> None:
        m = GeneralMemory(min_episodes=2, gradual_decay=False)
        for ep in (0, 1):
            m.practice("窗前明月光", "疑似地上霜", ep)
        m.consolidate()
        m.compress()
        self.assertEqual(len(m.verbatim), 0)
        self.assertFalse(m.reconsolidate("窗前明月光", "疑似地上霜", 100))
        self.assertTrue(m.reconsolidate("窗前明月光", "疑似地上霜", 101))
        self.assertEqual(m.query("窗前明月光").response, "疑似地上霜")


class TestUvicornStartupPersistence(unittest.TestCase):
    """Bug 1: uvicorn :app 直启(经 lifespan)必须挂上默认持久化。"""

    def test_lifespan_enables_default_persistence(self) -> None:
        old = api.state
        try:
            api.state = api._DEFAULT_STATE
            with TestClient(api.app) as client:
                self.assertEqual(api.state.memory_file, api.DEFAULT_MEMORY_FILE)
                stats = client.get("/v1/memories/stats").json()
                self.assertEqual(stats["persist_file"], str(api.DEFAULT_MEMORY_FILE))
        finally:
            api.state = old

    def test_lifespan_does_not_override_main_config(self) -> None:
        old = api.state
        try:
            api.state = api.build_state(memory_file=None)  # 模拟 --no-persist
            with TestClient(api.app):
                self.assertIsNone(api.state.memory_file)
        finally:
            api.state = old


class TestQueryStatsPersistence(unittest.TestCase):
    """Bug 4: 命中统计要落盘（节流触发 + 停服 flush）。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "m.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _state(self, every: int) -> api.State:
        s = api.build_state(memory_file=self.path, persist_query_every=every)
        for i in range(3):
            s.memory.practice("1+1", "2", i)
        s.memory.consolidate()
        s.persist()
        return s

    def test_throttled_persist(self) -> None:
        s = self._state(every=3)
        key = s.memory.encoder.key_of("1+1")
        for _ in range(3):
            s.query("1+1", 0.0)
        self.assertEqual(GeneralMemory.load(self.path).verbatim[key].hits, 3)

    def test_shutdown_flush(self) -> None:
        s = self._state(every=100)
        key = s.memory.encoder.key_of("1+1")
        for _ in range(2):
            s.query("1+1", 0.0)
        self.assertEqual(GeneralMemory.load(self.path).verbatim[key].hits, 0)
        old = api.state
        try:
            api.state = s
            with TestClient(api.app):
                pass
        finally:
            api.state = old
        self.assertEqual(GeneralMemory.load(self.path).verbatim[key].hits, 2)


class TestCachePruning(unittest.TestCase):
    """Bug 5: 缓存随固化/压缩清理，不再随练习无限增长。"""

    def test_consolidate_and_compress_prune_cache(self) -> None:
        m = GeneralMemory(min_episodes=1, gradual_decay=False)
        for i in range(5):
            m.practice("内容%d" % i, "答%d" % i, i)
        m.consolidate()
        self.assertEqual(len(m._feat_cache), 0)
        self.assertEqual(len(m._text_cache), 0)
        m.compress()
        self.assertEqual(len(m._feat_cache), 0)
        self.assertEqual(len(m.verbatim), 0)
        self.assertEqual(len(m.gists), 5)

    def test_pending_cache_survives_until_consolidated(self) -> None:
        m = GeneralMemory(min_episodes=3)
        for i in range(2):
            m.practice("未固化", "答案", i)
        # 未达门槛：pending 及其缓存必须保留，才能继续巩固
        self.assertIn(m.encoder.key_of("未固化"), m._feat_cache)
        m.practice("未固化", "答案", 2)
        m.consolidate()
        self.assertEqual(m.query("未固化").response, "答案")


if __name__ == "__main__":
    unittest.main(verbosity=2)
