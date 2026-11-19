"""OpenAI 兼容的 HTTP 服务：把通用内容记忆暴露成标准大模型流式 API。

端点（全部兼容 OpenAI 客户端的调用习惯）：
    GET  /v1/models                     模型列表
    POST /v1/chat/completions           对话；stream=true 走 SSE 流式
    POST /v1/memories                   练习入口（写路径），练完可巩固/压缩成要旨
    GET  /v1/memories/stats             当前记忆占用与巩固统计
    GET  /health                        健康检查

设计约束：
- 流式格式严格按 OpenAI：每个 chunk 是 `data: {json}\n\n`，
  先发一个带 role 的空 delta，再逐 token 发 content，最后 finish_reason=stop，
  以 `data: [DONE]\n\n` 收尾。非流式返回标准 chat.completion 对象。
- 准确性优先：记忆没命中（source=none）时，流式也返回空内容 + finish_reason=stop，
  不编造答案。查询的 source / weight / confident 放在 `usage` 扩展字段里供审计。
- 两级判定：联想激活（能不能想起候选）由记忆内核的 radius/threshold 决定，
  输出授权（够不够格当确定答案）由 confident 决定。默认非严格模式返回答案
  并附 confident 标记；请求带 `strict:true` 时，未授权的低置信联想一律返回空。
  执行类场景应开 strict，避免「想起来个模糊的」被当成确定指令执行。
- 记忆状态在进程内单实例，练习与查询共享。默认持久化到
  checkpoints/general_memory.json：启动时自动恢复，写操作后自动落盘；
  `--no-persist` 可退回纯内存模式。查询只更新命中统计，不每次写盘。
- token 切分按字符：中文没有空格分词，按字符流最自然，也和「逐字脱口而出」一致。

启动：
    python -m ultra_instinct.api --host 127.0.0.1 --port 8000
或：
    uvicorn ultra_instinct.api:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

try:
    from ultra_instinct.encoding import ContentEncoder
    from ultra_instinct.semantic import SemanticEncoder
    from ultra_instinct.forgetting import ForgettingProfile
    from ultra_instinct.general_memory import Answer, GeneralMemory
except ImportError:  # 直接跑本文件时
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ultra_instinct.encoding import ContentEncoder
    from ultra_instinct.semantic import SemanticEncoder
    from ultra_instinct.forgetting import ForgettingProfile
    from ultra_instinct.general_memory import Answer, GeneralMemory

MODEL_ID = "ultra-instinct"


# ---------------------------------------------------------------- 请求模型


class ChatMessage(BaseModel):
    role: str
    content: str | None = None


class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage] = Field(default_factory=list)
    stream: bool = False
    temperature: float | None = None     # 接受但不使用：背下来的答案不随机采样
    max_tokens: int | None = None
    now: float = 0.0                     # 记忆时间（天），用于遗忘/衰减
    # 本次是否只采纳达到输出授权门的答案。默认 False：返回答案 + confident
    # 标记，由调用方决定信不信（兼容旧行为）。执行类场景应设 True，
    # 低置信联想会返回空内容，避免「一本正经答错」。
    strict: bool = False


class MemoryPractice(BaseModel):
    content: str
    response: str
    episode_id: int
    now: float = 0.0


class PracticeRequest(BaseModel):
    items: list[MemoryPractice] = Field(default_factory=list)
    consolidate: bool = True
    min_episodes: int | None = None
    # 练完是否把相似逐字压成要旨。默认 False：压缩会真删逐字，是有损操作，
    # 不该在普通练习里悄悄发生。要测/用 gist 路径时显式打开。
    compress: bool = False


# ---------------------------------------------------------------- 服务状态


DEFAULT_MEMORY_FILE = (Path(__file__).resolve().parent.parent
                       / "checkpoints" / "general_memory.json")


@dataclass
class State:
    memory: GeneralMemory
    total_practices: int = 0
    total_queries: int = 0
    memory_file: Path | None = None
    last_saved: float | None = None
    # 查询会改 hits/last_used/uses，但每次查询都 fsync 会毁掉微秒级热路径。
    # 折中：每 persist_query_every 次查询节流落一次，停服时再 flush 一次。
    persist_query_every: int = 50
    _queries_since_save: int = 0

    def query(self, text: str, now: float) -> Answer:
        self.total_queries += 1
        answer = self.memory.query(text, now=now)
        # 命中统计的节流落盘；未配置路径时 persist() 直接返回 False。
        self._queries_since_save += 1
        if self._queries_since_save >= max(1, self.persist_query_every):
            self.persist()
        return answer

    def persist(self) -> bool:
        """写操作/节流点后落盘；未配置文件路径时什么都不做。"""
        if self.memory_file is None:
            return False
        self.memory.save(self.memory_file)
        self.last_saved = time.time()
        self._queries_since_save = 0
        return True


def build_state(enable_forgetting: bool = True,
                min_confidence: float | None = 0.6,
                memory_file: str | os.PathLike[str] | None = None,
                load_if_exists: bool = False,
                persist_query_every: int = 50,
                semantic_weights: str | os.PathLike[str] | None = None,
                gist_threshold: float = 0.55) -> State:
    """构造服务状态。

    memory_file 指定持久化文件；load_if_exists 为真且文件存在时从它恢复，
    否则新建空记忆。测试不传 memory_file，天然隔离、不碰磁盘。
    persist_query_every 命中统计的节流落盘间隔（查询次数）。
    semantic_weights 给出语义编码器 npz 时注入 SemanticEncoder；否则用
    默认 ContentEncoder（字符特征）。恢复与新建必须用同一编码器，否则
    存盘特征的维度/空间会与查询时错配。
    gist_threshold 相似判定门槛。字符场景默认 0.55；语义场景建议调到
    0.85 左右——语义向量下同义约 0.97、域外也有 0.65~0.79，低门槛会
    把无关输入误判成命中。
    """
    path = Path(memory_file) if memory_file is not None else None
    encoder = (SemanticEncoder.from_weights(str(semantic_weights))
               if semantic_weights else ContentEncoder())
    if path is not None and load_if_exists and path.exists():
        memory = GeneralMemory.load(path, encoder=encoder)
        memory.gist_threshold = float(gist_threshold)
    else:
        profile = ForgettingProfile() if enable_forgetting else None
        memory = GeneralMemory(encoder,
                               min_episodes=3,
                               min_confidence=min_confidence,
                               gist_threshold=gist_threshold,
                               forgetting=profile)
    return State(memory=memory, memory_file=path,
                 persist_query_every=persist_query_every)


state = build_state()
# 哨兵：模块加载时的初始 state。若 lifespan 触发时 state 仍是这个对象，
# 说明是 `uvicorn ultra_instinct.api:app` 直启（没经过 main()），需要补上
# 默认持久化配置。经 main() 或测试重新赋值过的 state 不受影响。
_DEFAULT_STATE = state

# uvicorn 直启（无 main）时的可选持久化配置：由环境变量提供，避免绕过 main
# 后丢失语义编码器等设置。main() 路径不依赖它。
_LIFESPAN_SEMANTIC_WEIGHTS = os.environ.get("UI_SEMANTIC_WEIGHTS") or None
_LIFESPAN_GIST_THRESHOLD = float(os.environ.get("UI_GIST_THRESHOLD", "0.55"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global state
    if state is _DEFAULT_STATE:
        state = build_state(memory_file=str(DEFAULT_MEMORY_FILE),
                            load_if_exists=True,
                            semantic_weights=_LIFESPAN_SEMANTIC_WEIGHTS,
                            gist_threshold=_LIFESPAN_GIST_THRESHOLD)
    yield
    # 正常停服时 flush 一次，保证不足节流阈值的命中统计也不丢。
    state.persist()


app = FastAPI(title="Ultra Instinct API", version="0.1.0",
              lifespan=lifespan)


# ---------------------------------------------------------------- 工具函数


def _user_text(messages: list[ChatMessage]) -> str:
    """取最后一条 user 消息作为查询输入；没有则报错。"""
    for m in reversed(messages):
        if m.role == "user" and m.content is not None:
            return m.content
    raise HTTPException(status_code=400, detail="messages 里没有 user 内容")


def _chunk(chunk_id: str, created: int, content: str | None,
           role: bool = False, finish_reason: str | None = None) -> dict:
    delta: dict[str, Any] = {}
    if role:
        delta["role"] = "assistant"
    if content:
        delta["content"] = content
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_ID,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }


def _sse(payload: dict | str) -> str:
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _usage(answer: Answer) -> dict:
    """把命中信息放进 usage 扩展字段，客户端可读，不破坏 OpenAI 结构。"""
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "source": answer.source,
        "weight": round(answer.weight, 6),
        "confident": answer.confident,
    }


# ---------------------------------------------------------------- 端点


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": MODEL_ID}


@app.get("/v1/models")
async def models() -> dict:
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": 0,
            "owned_by": "ultra-instinct",
        }],
    }


@app.post("/v1/memories")
async def practice(req: PracticeRequest) -> dict:
    for it in req.items:
        state.memory.practice(it.content, it.response, it.episode_id, now=it.now)
        state.total_practices += 1
    promoted = 0
    if req.consolidate:
        promoted = state.memory.consolidate(req.min_episodes)
    compressed: dict = {}
    if req.compress:
        compressed = state.memory.compress()
    # 写操作后落盘。查询只更新命中统计（hits/last_used），不每次写盘——
    # 那是热路径，落盘交给下次写操作时一并保存。
    saved = state.persist()
    return {
        "practiced": len(req.items),
        "promoted": promoted,
        "verbatim": len(state.memory.verbatim),
        "compressed": compressed,
        "saved": saved,
    }


@app.get("/v1/memories/stats")
async def stats() -> dict:
    return {
        "footprint": state.memory.footprint,
        "verbatim": len(state.memory.verbatim),
        "gists": len(state.memory.gists),
        "episodic": len(state.memory.episodic),
        "total_practices": state.total_practices,
        "total_queries": state.total_queries,
        "forgetting_enabled": state.memory.forgetting is not None,
        "persist_file": (str(state.memory_file)
                         if state.memory_file is not None else None),
        "last_saved": state.last_saved,
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if req.model != MODEL_ID:
        raise HTTPException(status_code=404,
                            detail=f"未知模型 {req.model!r}，可用：{MODEL_ID}")

    text = _user_text(req.messages)
    answer = state.query(text, req.now)
    # strict：把未授权的低置信联想视作没命中。source 也改写成 none，
    # 让调用方看到的是「不知道」而不是一个不该信的答案。非 strict 保留
    # 答案 + confident 标记，由调用方自己决定。
    if req.strict and not answer.confident:
        answer = Answer(None, 0.0, "none", None, False)
    content = answer.response or ""
    if req.max_tokens is not None and req.max_tokens >= 0:
        content = content[:req.max_tokens]

    created = int(time.time())
    chunk_id = "chatcmpl-" + uuid.uuid4().hex[:12]

    if not req.stream:
        return {
            "id": chunk_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": _usage(answer),
        }

    async def token_stream() -> AsyncIterator[str]:
        # OpenAI 常见顺序：先发一个只有 role 的空 delta
        yield _sse(_chunk(chunk_id, created, None, role=True))
        for ch in content:
            yield _sse(_chunk(chunk_id, created, ch))
        yield _sse(_chunk(chunk_id, created, None, finish_reason="stop"))
        yield _sse("[DONE]")

    return StreamingResponse(
        token_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # 关掉 nginx 缓冲，保证真流式
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ultra Instinct OpenAI 兼容 API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-forgetting", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.6,
                        help="输出授权门 0~1；0 表示不设门")
    parser.add_argument("--memory-file", default=str(DEFAULT_MEMORY_FILE),
                        help="记忆持久化文件；启动时若存在则自动恢复")
    parser.add_argument("--no-persist", action="store_true",
                        help="关闭持久化（纯内存，重启即丢）")
    parser.add_argument("--semantic-weights", default=None,
                        help="语义编码器权重 npz；不传则用字符编码器")
    parser.add_argument("--gist-threshold", type=float, default=0.55,
                        help="相似判定门槛；语义场景建议 0.85 挡住域外误命中")
    args = parser.parse_args()

    global state, _LIFESPAN_SEMANTIC_WEIGHTS, _LIFESPAN_GIST_THRESHOLD
    _LIFESPAN_SEMANTIC_WEIGHTS = args.semantic_weights
    _LIFESPAN_GIST_THRESHOLD = args.gist_threshold
    state = build_state(enable_forgetting=not args.no_forgetting,
                        min_confidence=args.min_confidence,
                        memory_file=None if args.no_persist else args.memory_file,
                        load_if_exists=not args.no_persist,
                        semantic_weights=args.semantic_weights,
                        gist_threshold=args.gist_threshold)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
