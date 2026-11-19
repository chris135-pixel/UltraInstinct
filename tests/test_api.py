"""HTTP API 测试：用 FastAPI TestClient，不需要起外部服务。"""

from __future__ import annotations

import json
import unittest

from fastapi.testclient import TestClient

import ultra_instinct.api as api


class TestAPI(unittest.TestCase):
    def setUp(self) -> None:
        # 每条用例独立状态，避免相互污染
        api.state = api.build_state()
        # 这批用例测 API 压缩/门控本身，锚定旧语义（练完立即压缩生效）；
        # 时间驱动的渐进退化由 test_gradual_decay 单独验证。
        api.state.memory.gradual_decay = False
        self.client = TestClient(api.app)

    def test_health(self) -> None:
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_models(self) -> None:
        data = self.client.get("/v1/models").json()["data"]
        self.assertEqual(data[0]["id"], api.MODEL_ID)

    def test_practice_and_nonstream(self) -> None:
        items = [{"content": "1+1", "response": "2",
                  "episode_id": i, "now": i} for i in range(3)]
        r = self.client.post("/v1/memories", json={"items": items})
        self.assertEqual(r.json()["promoted"], 1)

        r = self.client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "1+1"}],
            "stream": False,
        })
        body = r.json()
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["message"]["content"], "2")
        self.assertEqual(body["usage"]["source"], "verbatim")

    def test_stream_sse(self) -> None:
        items = [{"content": "窗前明月光", "response": "疑似地上霜",
                  "episode_id": i, "now": i} for i in range(3)]
        self.client.post("/v1/memories", json={"items": items})

        with self.client.stream("POST", "/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "窗前明月光"}],
            "stream": True,
        }) as r:
            self.assertIn("text/event-stream", r.headers["content-type"])
            chunks: list[str] = []
            finished = False
            done = False
            for raw in r.iter_lines():
                if not raw.startswith("data: "):
                    continue
                payload = raw[6:]
                if payload == "[DONE]":
                    done = True
                    break
                obj = json.loads(payload)
                self.assertEqual(obj["object"], "chat.completion.chunk")
                delta = obj["choices"][0]["delta"]
                if "content" in delta:
                    chunks.append(delta["content"])
                if obj["choices"][0]["finish_reason"] == "stop":
                    finished = True
            self.assertTrue(done)
            self.assertTrue(finished)
            self.assertEqual("".join(chunks), "疑似地上霜")

    def test_unknown_not_answered(self) -> None:
        r = self.client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "没练过的"}],
            "stream": False,
        })
        self.assertEqual(r.json()["choices"][0]["message"]["content"], "")
        self.assertEqual(r.json()["usage"]["source"], "none")

    def test_practice_compress_switch(self) -> None:
        """compress=true 才压缩；默认不压缩（有损操作不悄悄发生）。"""
        items = []
        ep = 0
        for content, resp in (("窗前明月光", "疑似地上霜"),
                              ("床前明月光", "疑似地上霜"),
                              ("月光明窗前", "疑似地上霜")):
            for _ in range(3):
                items.append({"content": content, "response": resp,
                              "episode_id": ep, "now": ep})
                ep += 1
        # 默认：只巩固，不压缩
        r1 = self.client.post("/v1/memories", json={"items": items}).json()
        self.assertEqual(r1["compressed"], {})
        self.assertGreater(r1["verbatim"], 0)
        # 显式开 compress：逐字被合并成 gist
        r2 = self.client.post("/v1/memories", json={
            "items": [], "consolidate": False, "compress": True}).json()
        self.assertGreater(r2["compressed"]["merged"], 0)

    def test_usage_exposes_confident(self) -> None:
        items = [{"content": "1+1", "response": "2",
                  "episode_id": i, "now": i} for i in range(3)]
        self.client.post("/v1/memories", json={"items": items})
        r = self.client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "1+1"}],
        }).json()
        self.assertIn("confident", r["usage"])
        self.assertTrue(r["usage"]["confident"])

    def test_strict_drops_low_confidence(self) -> None:
        """strict=true 时低置信命中返回空；verbatim 不受影响。"""
        # 造一个低 coherence 的要旨簇（门 0.6）
        items = []
        ep = 0
        for content, resp in (("窗前明月光", "疑似地上霜"),
                              ("床前明月光", "疑似地上霜"),
                              ("月光明窗前", "低头思故乡")):
            for _ in range(3):
                items.append({"content": content, "response": resp,
                              "episode_id": ep, "now": ep})
                ep += 1
        items += [{"content": "1+1", "response": "2",
                   "episode_id": ep + i, "now": ep + i} for i in range(3)]
        self.client.post("/v1/memories", json={"items": items})
        # 用压缩制造低置信要旨
        api.state.memory.compress()
        # 非 strict：返回带 confident 标记
        loose = self.client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "床前明月光"}],
        }).json()
        self.assertFalse(loose["usage"]["confident"])
        # strict：低置信 -> 空 + source=none
        strict = self.client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "床前明月光"}],
            "strict": True,
        }).json()
        self.assertEqual(strict["choices"][0]["message"]["content"], "")
        self.assertEqual(strict["usage"]["source"], "none")
        # strict 下 verbatim 仍正常
        verbatim = self.client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "1+1"}],
            "strict": True,
        }).json()
        self.assertEqual(verbatim["choices"][0]["message"]["content"], "2")
        self.assertTrue(verbatim["usage"]["confident"])

    def test_strict_stream_drops_low_confidence(self) -> None:
        """流式与非流式共用裁决点：strict 下低置信也没有任何 content chunk。"""
        items = []
        ep = 0
        for content, resp in (("窗前明月光", "疑似地上霜"),
                              ("床前明月光", "疑似地上霜"),
                              ("月光明窗前", "低头思故乡")):
            for _ in range(3):
                items.append({"content": content, "response": resp,
                              "episode_id": ep, "now": ep})
                ep += 1
        self.client.post("/v1/memories", json={"items": items})
        api.state.memory.compress()
        with self.client.stream("POST", "/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "月光明窗前"}],
            "stream": True,
            "strict": True,
        }) as r:
            contents = []
            for raw in r.iter_lines():
                if not raw.startswith("data: ") or raw == "data: [DONE]":
                    continue
                delta = json.loads(raw[6:])["choices"][0]["delta"]
                if "content" in delta:
                    contents.append(delta["content"])
            self.assertEqual("".join(contents), "")

    def test_wrong_model_404(self) -> None:
        r = self.client.post("/v1/chat/completions", json={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "x"}],
        })
        self.assertEqual(r.status_code, 404)

    def test_max_tokens(self) -> None:
        items = [{"content": "窗前明月光", "response": "疑似地上霜",
                  "episode_id": i, "now": i} for i in range(3)]
        self.client.post("/v1/memories", json={"items": items})
        r = self.client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "窗前明月光"}],
            "max_tokens": 2,
        })
        self.assertEqual(r.json()["choices"][0]["message"]["content"], "疑似")


if __name__ == "__main__":
    unittest.main(verbosity=2)
