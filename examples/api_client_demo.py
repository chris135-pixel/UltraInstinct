"""HTTP API 客户端演示：非流式 + SSE 流式，兼容 OpenAI 调用习惯。

先启动服务：
    python -m ultra_instinct.api --host 127.0.0.1 --port 8000
再跑：
    python examples/api_client_demo.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

BASE = "http://127.0.0.1:8000"


def post(path: str, payload: dict) -> dict:
    r = httpx.post(BASE + path, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def line(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def main() -> None:
    line("[1] 练内容进记忆（跨 3 个 episode 才固化）")
    items = [
        {"content": "1+1", "response": "2", "episode_id": 0, "now": 0},
        {"content": "1+1", "response": "2", "episode_id": 1, "now": 1},
        {"content": "1+1", "response": "2", "episode_id": 2, "now": 2},
        {"content": "窗前明月光", "response": "疑似地上霜", "episode_id": 3, "now": 3},
        {"content": "窗前明月光", "response": "疑似地上霜", "episode_id": 4, "now": 4},
        {"content": "窗前明月光", "response": "疑似地上霜", "episode_id": 5, "now": 5},
    ]
    print(post("/v1/memories", {"items": items, "consolidate": True}))

    line("[2] 非流式（stream=false），标准 chat.completion 结构")
    r = post("/v1/chat/completions", {
        "model": "ultra-instinct",
        "messages": [{"role": "user", "content": "1+1"}],
        "stream": False,
    })
    print("  答案:", r["choices"][0]["message"]["content"])
    print("  usage(扩展):", r["usage"])

    line("[3] 流式（stream=true），逐块读 SSE")
    with httpx.stream("POST", BASE + "/v1/chat/completions", json={
        "model": "ultra-instinct",
        "messages": [{"role": "user", "content": "窗前明月光"}],
        "stream": True,
    }, timeout=30) as resp:
        resp.raise_for_status()
        print("  Content-Type:", resp.headers.get("content-type"))
        pieces: list[str] = []
        n_chunk = 0
        for raw in resp.iter_lines():
            if not raw.startswith("data: "):
                continue
            data = raw[6:]
            if data == "[DONE]":
                print("  收到 [DONE]")
                break
            obj = json.loads(data)
            n_chunk += 1
            delta = obj["choices"][0]["delta"]
            if "content" in delta:
                pieces.append(delta["content"])
        print(f"  收到 {n_chunk} 个 chunk，拼出的答案：{''.join(pieces)}")

    line("[4] 没练过的问题：不乱答")
    r = post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "今天天气怎么样"}],
        "stream": False,
    })
    print("  答案:", repr(r["choices"][0]["message"]["content"]))
    print("  source:", r["usage"]["source"])

    line("[5] 记忆统计")
    stats = httpx.get(BASE + "/v1/memories/stats", timeout=10).json()
    print(" ", json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
