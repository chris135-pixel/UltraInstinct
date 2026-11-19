"""端到端指标报告：训练几条简单数据 → 量出六组真实指标。

不同于别的 benchmark（只测底层的一件事），这个脚本回答一个完整问题：
「真的拿几条内容练一练，这套系统在准确性、速度、容量、遗忘上到底表现如何？」

六组指标，全部实测，不填估计值：
    1. 训练与巩固   —— 练习条数 / 耗时 / 提升(consolidate)数 / 逐字条目
    2. 准确性       —— 练过的命中率、没练过的拒答率、错字容错
    3. 延迟         —— 内核 query(逐字/要旨/未命中)、HTTP 端到端、流式首包
    4. 记忆占用     —— footprint（逐字/要旨/浅记忆参数量）
    5. 遗忘         —— 不同复杂度要旨的保持率矩阵 + 半衰期（论文参数）
    6. 流式         —— SSE chunk 数 / 拼接正确性 / [DONE] / 逐字吐出延迟

数据经由真实 HTTP 服务（POST /v1/memories、/v1/chat/completions）写入和查询，
内核微基准用独立实例直接调 GeneralMemory，避免网络栈掩盖微秒级差异。

前置：先启动服务
    python -m ultra_instinct.api --host 127.0.0.1 --port 8000
运行：
    python benchmarks/bench_metrics.py
输出：
    终端 Markdown 报告 + benchmarks/metrics_report.json
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from ultra_instinct import ContentEncoder, GeneralMemory  # noqa: E402
from ultra_instinct import forgetting as F  # noqa: E402

BASE = os.environ.get("UI_BASE_URL", "http://127.0.0.1:8000")
MODEL = "ultra-instinct"
CLIENT = httpx.Client(base_url=BASE, timeout=30.0)

# ---------------------------------------------------------------- 测试数据

DATA = [
    ("1+1", "2"),
    ("2+2", "4"),
    ("3*7", "21"),
    ("窗前明月光", "疑似地上霜"),
    ("春眠不觉晓", "处处闻啼鸟"),
    ("举头望明月", "低头思故乡"),
]
UNSEEN = ["今天天气怎么样", "帮我写一段 Python", "9 月 11 日发生了什么"]
TYPO_QUERY = "窗前明月"          # 练过「窗前明月光」，去掉尾字测模糊容错
MIN_EPISODES = 3


def hr(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def table(headers: list[str], rows: list[list[str]]) -> None:
    w = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            w[i] = max(w[i], len(c))
    print("| " + " | ".join(h.ljust(w[i]) for i, h in enumerate(headers)) + " |")
    print("|" + "|".join("-" * (x + 2) for x in w) + "|")
    for r in rows:
        print("| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(r)) + " |")


def post(path: str, payload: dict) -> dict:
    r = CLIENT.post(path, json=payload)
    r.raise_for_status()
    return r.json()


def chat(content: str, stream: bool = False, now: float = 0.0) -> dict:
    return post("/v1/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "stream": stream,
        "now": now,
    })


def answer_of(resp: dict) -> tuple[str, str, float]:
    ch = resp["choices"][0]
    u = resp.get("usage", {})
    return ch["message"]["content"] or "", u.get("source", "?"), float(u.get("weight", 0.0))


# ---------------------------------------------------------------- 微基准

def bench_us(fn, iters: int = 1500, warmup: int = 300) -> dict[str, float]:
    """稳态单步延迟；逐次 perf_counter_ns，单位微秒。numpy 查询约几十微秒，
    QPC 本身的 ~0.1µs 开销可忽略。"""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        ts.append(time.perf_counter_ns() - t0)
    ts.sort()
    return {
        "p50_us": round(ts[len(ts) // 2] / 1000.0, 1),
        "p99_us": round(ts[int(len(ts) * 0.99)] / 1000.0, 1),
        "mean_us": round(statistics.fmean(ts) / 1000.0, 1),
    }


def bench_http(fn, iters: int = 60, warmup: int = 5) -> dict[str, float]:
    """HTTP 端到端延迟（含网络栈 + uvicorn + JSON 解析），单位毫秒。"""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        ts.append((time.perf_counter_ns() - t0) / 1e6)
    ts.sort()
    return {
        "p50_ms": round(ts[len(ts) // 2], 3),
        "p99_ms": round(ts[int(len(ts) * 0.99)], 3),
        "mean_ms": round(statistics.fmean(ts), 3),
    }


# ---------------------------------------------------------------- 主流程

def main() -> None:
    report: dict = {}

    # 先探活
    try:
        CLIENT.get("/health").raise_for_status()
    except Exception as exc:
        print(f"[错误] 连不上服务 {BASE}/health：{exc}")
        print("请先启动：python -m ultra_instinct.api --host 127.0.0.1 --port 8000")
        sys.exit(1)

    # ---------- [1] 训练与巩固 ----------
    hr("[1] 训练与巩固")
    items = []
    ep = 0
    for content, resp in DATA:
        for _ in range(MIN_EPISODES):
            items.append({"content": content, "response": resp,
                          "episode_id": ep, "now": float(ep)})
            ep += 1
    t0 = time.perf_counter_ns()
    train = post("/v1/memories", {"items": items, "consolidate": True,
                                  "min_episodes": MIN_EPISODES})
    train_ms = (time.perf_counter_ns() - t0) / 1e6
    train_row = {
        "练习条数": len(items),
        "耗时_ms": round(train_ms, 2),
        "每条练习_us": round(train_ms * 1000 / len(items), 1),
        "提升为逐字": train["promoted"],
        "逐字条目": train["verbatim"],
    }
    report["train"] = train_row
    table(["指标", "实测值"], [[k, str(v)] for k, v in train_row.items()])

    # ---------- [2] 准确性 ----------
    hr("[2] 准确性")
    hit = 0
    acc_rows = []
    for content, expected in DATA:
        a, src, w = answer_of(chat(content))
        ok = (a == expected and src == "verbatim")
        hit += int(ok)
        acc_rows.append([content, expected, a or "(空)", src, f"{w:.2f}",
                         "正确" if ok else "错"])
    hit_rate = hit / len(DATA) * 100

    refuse = 0
    for q in UNSEEN:
        a, src, w = answer_of(chat(q))
        refuse += int(a == "" and src == "none")
    refuse_rate = refuse / len(UNSEEN) * 100

    ta, tsrc, tw = answer_of(chat(TYPO_QUERY))
    typo_ok = (ta == "疑似地上霜")

    table(["输入", "应为", "实得", "来源", "权重", "判定"], acc_rows)
    acc_summary = {
        "练过命中率": f"{hit}/{len(DATA)} = {hit_rate:.0f}%",
        "未练拒答率": f"{refuse}/{len(UNSEEN)} = {refuse_rate:.0f}%",
        "错字容错": f"'{TYPO_QUERY}' → {ta or '(空)'} [{tsrc}, w={tw:.2f}] "
                    f"{'命中' if typo_ok else '未命中'}",
    }
    report["accuracy"] = {
        "hit": f"{hit}/{len(DATA)}", "hit_rate": round(hit_rate, 1),
        "refuse": f"{refuse}/{len(UNSEEN)}", "refuse_rate": round(refuse_rate, 1),
        "typo_query": TYPO_QUERY, "typo_answer": ta,
        "typo_source": tsrc, "typo_weight": round(tw, 3), "typo_ok": typo_ok,
    }
    print()
    table(["指标", "实测值"], [[k, v] for k, v in acc_summary.items()])

    # ---------- [3] 延迟 ----------
    hr("[3] 延迟")

    # 内核级：独立实例，避免网络栈
    km = GeneralMemory(ContentEncoder(), min_episodes=2)
    for c, r in DATA:
        for e in range(2):
            km.practice(c, r, e)
    km.consolidate()
    lat_verbatim = bench_us(lambda: km.query("1+1"))
    lat_none = bench_us(lambda: km.query("完全没练过的随机句子xyz"))

    # 造一个 gist 用于测要旨路径。渐进退化默认开启，now=0 时细节完整度
    # 仍为 1.0，compress() 不会沉淀；这里走时间退化入口 decay(now=30)。
    km2 = GeneralMemory(ContentEncoder(), min_episodes=2)
    for c in ["窗前明月光", "窗前明月光。", "窗前明月光！"]:
        for e in range(2):
            km2.practice(c, "疑似地上霜", e)
    km2.consolidate()
    km2.decay(now=30)
    if km2.gists:
        lat_gist = bench_us(lambda: km2.query("窗前明月光"))
        gist_cell = f"{lat_gist['p50_us']:.1f} / {lat_gist['p99_us']:.1f} µs"
    else:
        gist_cell = "N/A（未形成要旨）"
        lat_gist = None

    kernel_rows = [
        ["内核 · 逐字命中", f"{lat_verbatim['p50_us']:.1f} / {lat_verbatim['p99_us']:.1f} µs"],
        ["内核 · 要旨命中", gist_cell],
        ["内核 · 未命中", f"{lat_none['p50_us']:.1f} / {lat_none['p99_us']:.1f} µs"],
    ]
    print("内核级（直接调 GeneralMemory.query，p50 / p99）：")
    table(["路径", "延迟"], kernel_rows)

    # HTTP 级：真实网络请求
    lat_http = bench_http(lambda: chat("1+1"))
    lat_http_none = bench_http(lambda: chat("完全没练过的随机句子xyz"))
    http_rows = [
        ["HTTP · 逐字命中", f"{lat_http['p50_ms']:.3f} / {lat_http['p99_ms']:.3f} ms"],
        ["HTTP · 未命中", f"{lat_http_none['p50_ms']:.3f} / {lat_http_none['p99_ms']:.3f} ms"],
    ]
    print("\nHTTP 端到端（真实请求，含网络栈，p50 / p99）：")
    table(["路径", "延迟"], http_rows)

    report["latency"] = {
        "kernel_verbatim": lat_verbatim,
        "kernel_gist": lat_gist,
        "kernel_none": lat_none,
        "http_verbatim": lat_http,
        "http_none": lat_http_none,
    }

    # ---------- [4] 记忆占用 ----------
    hr("[4] 记忆占用")
    stats = CLIENT.get("/v1/memories/stats").json()
    fp = stats["footprint"]
    fp_rows = [[k, str(v)] for k, v in fp.items()]
    table(["类别", "等效参数量"], fp_rows)
    report["footprint"] = fp

    # ---------- [5] 遗忘 ----------
    hr("[5] 遗忘（论文参数：复杂度高 → 先忘）")
    ages = [("1 小时", 1 / 24), ("1 天", 1), ("7 天", 7), ("30 天", 30), ("1 年", 365)]
    compl = [("高(细节多) c=1.0", 1.0), ("中 c=0.6", 0.6), ("低(核心要旨) c=0.2", 0.2)]
    rows = [["年龄"] + [c for c, _ in compl]]
    for label, age in ages:
        rows.append([label] + [f"{F.retention(age, c) * 100:.1f}%" for _, c in compl])
    table(rows[0], rows[1:])
    half_rows = [[label, f"{F.halflife_of(c):.2f} 天"] for label, c in compl]
    print("\n混合保持率降到 50% 的时间：")
    table(["复杂度", "半衰期"], half_rows)
    report["forgetting"] = {
        "ages": [{"label": l, "days": d} for l, d in ages],
        "matrix": {label: [round(F.retention(d, c) * 100, 1) for _, c in compl]
                   for label, d in ages},
        "halflife_days": {label: round(F.halflife_of(c), 2) for label, c in compl},
    }

    # ---------- [6] 流式 ----------
    hr("[6] 流式（SSE，逐字吐出）")
    t0 = time.perf_counter_ns()
    first_chunk_ms = None
    chunks = 0
    pieces: list[str] = []
    got_done = False
    content_type = ""
    with CLIENT.stream("POST", "/v1/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "窗前明月光"}],
        "stream": True,
    }) as resp:
        content_type = resp.headers.get("content-type", "")
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw.startswith("data: "):
                continue
            data = raw[6:]
            if data == "[DONE]":
                got_done = True
                break
            obj = json.loads(data)
            chunks += 1
            if first_chunk_ms is None:
                first_chunk_ms = (time.perf_counter_ns() - t0) / 1e6
            delta = obj["choices"][0]["delta"]
            if "content" in delta:
                pieces.append(delta["content"])
    total_ms = (time.perf_counter_ns() - t0) / 1e6
    stitched = "".join(pieces)
    stream_rows = [
        ["Content-Type", content_type],
        ["chunk 数", str(chunks)],
        ["首包延迟", f"{first_chunk_ms:.3f} ms" if first_chunk_ms else "N/A"],
        ["总延迟", f"{total_ms:.3f} ms"],
        ["拼接答案", stitched],
        ["答案正确", "是" if stitched == "疑似地上霜" else "否"],
        ["收到 [DONE]", "是" if got_done else "否"],
    ]
    table(["指标", "实测值"], stream_rows)
    report["stream"] = {
        "content_type": content_type, "chunks": chunks,
        "first_chunk_ms": round(first_chunk_ms, 3) if first_chunk_ms else None,
        "total_ms": round(total_ms, 3), "answer": stitched,
        "correct": stitched == "疑似地上霜", "done": got_done,
    }

    # ---------- 落盘 ----------
    out = ROOT / "benchmarks" / "metrics_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    hr(f"完成 · 报告已写入 {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
