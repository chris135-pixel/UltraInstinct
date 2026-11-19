"""渐进退化（短期→长期保留细节→随时间不用才渐退成要旨）单元测试。

核心区分：旧的 compress 是"手动一步删光细节"，这验证的是"时间驱动的
连续退化 + 用进保鲜 + 再巩固回满"，且默认关闭时旧行为分毫不变。
"""

from __future__ import annotations

import os
import tempfile
import unittest

from ultra_instinct.general_memory import GeneralMemory


def build(gradual: bool, **kw) -> GeneralMemory:
    args = dict(min_episodes=1, complexity_scale=1.0, tau=1.0, alpha=1.0,
                gradual_decay=gradual, detail_threshold=0.5)
    args.update(kw)
    return GeneralMemory(**args)


def drill_cluster(m: GeneralMemory, n: int = 3) -> None:
    texts = ("窗前明月光", "床前明月光", "窗前看月光", "明月照窗前")[:n]
    for i, s in enumerate(texts):
        m.practice(s, "疑似地上霜", episode_id=i, now=0.0)
    m.consolidate()


class TestDefaultEnabled(unittest.TestCase):
    """新契约：渐进退化默认开启（对应人脑，无需显式打开）。"""

    def test_default_is_on(self) -> None:
        self.assertTrue(GeneralMemory().gradual_decay)
        self.assertEqual(GeneralMemory().detail_threshold, 0.5)

    def test_default_compress_keeps_fresh_detail(self) -> None:
        """默认下练完立即 compress 不会一步抹掉新鲜细节。"""
        m = GeneralMemory(min_episodes=1, tau=1.0, alpha=1.0)
        drill_cluster(m)
        st = m.compress(now=0.0)
        self.assertEqual(st["merged"], 0)          # 细节完整 → 不合并
        self.assertEqual(len(m.verbatim), 3)       # 逐字原样保留


class TestDefaultUnchanged(unittest.TestCase):
    """显式关闭渐进退化时：旧的全部行为不变。"""

    def test_detail_always_one_when_off(self) -> None:
        m = build(False)
        drill_cluster(m)
        for e in m.verbatim.values():
            self.assertEqual(m.detail(e, 10_000.0), 1.0)

    def test_compress_immediately_promotes_all(self) -> None:
        m = build(False)
        drill_cluster(m)
        st = m.compress(now=0.0)
        self.assertGreater(st["merged"], 0)
        self.assertEqual(len(m.verbatim), 0)   # 一步删光，旧行为

    def test_decay_is_noop_when_off(self) -> None:
        m = build(False)
        drill_cluster(m)
        before = len(m.verbatim)
        st = m.decay(now=10_000.0)
        self.assertFalse(st["enabled"])
        self.assertEqual(len(m.verbatim), before)


class TestContinuousDecay(unittest.TestCase):
    """开启后：细节随时间连续褪色。"""

    def test_detail_monotone_decreasing(self) -> None:
        m = build(True)
        drill_cluster(m, n=1)
        e = list(m.verbatim.values())[0]
        vals = [m.detail(e, float(t)) for t in (0, 1, 3, 7, 30)]
        for a, b in zip(vals, vals[1:]):
            self.assertGreater(a, b)
        self.assertAlmostEqual(vals[0], 1.0)

    def test_fresh_not_promoted_stale_promoted(self) -> None:
        m = build(True)
        drill_cluster(m)
        self.assertEqual(len(m.gists), 0)
        st_fresh = m.decay(now=0.0)
        self.assertEqual(st_fresh["entries_removed"], 0)
        self.assertEqual(len(m.gists), 0)
        st_stale = m.decay(now=100.0)
        self.assertGreater(st_stale["entries_removed"], 0)
        self.assertGreater(len(m.gists), 0)

    def test_decay_time_anchor_is_last_use_not_born(self) -> None:
        """born 曾被存成序号而非时间；这里锚定真实练习时间。"""
        m = build(True)
        m.practice("窗前明月光", "疑似地上霜", 0, now=50.0)
        m.consolidate()
        e = list(m.verbatim.values())[0]
        self.assertEqual(e.born, 50.0)
        self.assertEqual(e.last_used, 50.0)
        self.assertAlmostEqual(m.detail(e, 50.0), 1.0)


class TestUseRefreshesDetail(unittest.TestCase):
    """用进：命中/再巩固刷新 last_used，细节回满、不被沉淀。"""

    def test_query_refreshes_and_survives_decay(self) -> None:
        m = build(True)
        drill_cluster(m, n=2)
        used, idle = "窗前明月光", "床前明月光"
        for t in range(1, 11):
            m.query(used, now=float(t))   # 反复使用一条
        m.decay(now=10.0)
        remaining = [e.enc.text for e in m.verbatim.values()]
        self.assertIn(used, remaining)      # 常用的留下完整细节
        self.assertNotIn(idle, remaining)   # 不用的沉淀成要旨

    def test_reconsolidation_restores_detail(self) -> None:
        m = build(True)
        drill_cluster(m, n=1)
        e = list(m.verbatim.values())[0]
        self.assertLess(m.detail(e, 10.0), 0.5)
        m.reconsolidate("窗前明月光", "疑似地上霜", 99, now=10.0)
        e2 = list(m.verbatim.values())[0]
        self.assertEqual(e2.last_used, 10.0)
        self.assertAlmostEqual(m.detail(e2, 10.0), 1.0)


class TestPersistence(unittest.TestCase):
    def test_roundtrip_keeps_gradual_state(self) -> None:
        m = build(True, detail_threshold=0.4)
        drill_cluster(m, n=1)
        m.query("窗前明月光", now=7.0)   # 制造非零 last_used
        d = tempfile.mkdtemp()
        p = os.path.join(d, "m.json")
        m.save(p)
        r = GeneralMemory.load(p)
        self.assertTrue(r.gradual_decay)
        self.assertAlmostEqual(r.detail_threshold, 0.4)
        e = list(r.verbatim.values())[0]
        self.assertEqual(e.last_used, 7.0)
        self.assertAlmostEqual(r.detail(e, 7.0), 1.0)

    def test_load_old_file_without_new_fields(self) -> None:
        """没有新字段的旧文件仍能加载，走默认开启（新默认）。"""
        m = GeneralMemory(min_episodes=1)
        drill_cluster(m, n=1)
        d = tempfile.mkdtemp()
        p = os.path.join(d, "old.json")
        m.save(p)
        import json
        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
        doc["config"].pop("gradual_decay", None)
        doc["config"].pop("detail_threshold", None)
        for item in doc["pending"]:
            item.pop("time", None)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
        r = GeneralMemory.load(p)
        self.assertTrue(r.gradual_decay)          # 缺省字段 → 新默认开启
        self.assertEqual(r.detail_threshold, 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
