"""真实 HTTP 验证输出授权门：练数据 → 查询 → 默认 vs strict 对比。

跑法：先启动服务，再 `python examples/confidence_gate_client.py`
"""
from __future__ import annotations

import json
import urllib.request

BASE = "http://127.0.0.1:8000"


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def line(title: str) -> None:
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


def show(label: str, body: dict) -> None:
    ch = body["choices"][0]
    content = (ch.get("message") or ch.get("delta") or {}).get("content", "")
    u = body["usage"]
    print(f"  {label:<28} content={content!r:<18} "
          f"source={u['source']:<9} weight={u['weight']:<7} confident={u['confident']}")


def main() -> None:
    line("1. 练数据：verbatim 1 条 + 低 coherence 要旨簇 3 条")
    items: list[dict] = []
    ep = 0
    plan = [("1+1", "2"),                       # 练熟 → verbatim
            ("窗前明月光", "疑似地上霜"),
            ("床前明月光", "疑似地上霜"),
            ("月光明窗前", "低头思故乡")]        # 冲突答案 → 拉低 coherence
    for content, resp in plan:
        for _ in range(3):
            items.append({"content": content, "response": resp,
                          "episode_id": ep, "now": float(ep)})
            ep += 1
    # compress=true 让服务端真把三条相似输入合成一个要旨（coherence=2/3）
    r = post("/v1/memories", {"items": items, "consolidate": True,
                              "compress": True})
    print("  练习返回:", json.dumps(r, ensure_ascii=False))
    print("  压缩返回:", json.dumps(r["compressed"], ensure_ascii=False))

    line("2. 默认模式（strict=false）：低置信仍返回原文，只打标记")
    for q in ("1+1", "窗前明月光", "床前明月光", "月光明窗前"):
        show(q, post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": q}]}))

    line("3. strict 模式：低置信一律当没命中，verbatim 不受影响")
    for q in ("1+1", "窗前明月光", "床前明月光", "月光明窗前"):
        show(q + " [strict]", post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": q}], "strict": True}))

    line("4. 没练过的：默认/strict 都一样拒答")
    for strict in (False, True):
        show(f"今天天气 [strict={strict}]", post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "今天天气怎么样"}],
            "strict": strict}))

    print("\n完成。")


if __name__ == "__main__":
    main()
