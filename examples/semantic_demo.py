"""语义编码器演示：本地自写网络 + KV 映射 → 换说法也能命中同一答案。

跑法：  python examples/semantic_demo.py

真跑四件事：
    1. 未训练的编码器 = 纯字符特征（退化可用），与 ContentEncoder 行为一致
    2. 用 KV 映射（同一答案的多种问法）对比训练一个小网络
    3. 训练后：练过的问法走逐字；没练过的同义问法走语义命中同一答案
    4. 精确路径仍在：一字不差的输入仍是 verbatim、微秒级
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultra_instinct import GeneralMemory
from ultra_instinct.encoding import ContentEncoder
from ultra_instinct.net import MLP
from ultra_instinct.semantic import SemanticEncoder
from ultra_instinct.training.semantic_train import load_pairs, train_semantic


def line(title: str = "") -> None:
    print("\n" + "=" * 64)
    if title:
        print(title)
        print("=" * 64)


def drill(mem: GeneralMemory, q: str, a: str, n: int, base: int) -> None:
    for k in range(n):
        mem.practice(q, a, base + k, now=base + k)
    mem.consolidate()


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    kv_path = os.path.join(here, "kv_example.json")
    pairs = load_pairs(kv_path)

    line("[1] 未训练的编码器 = 纯字符特征，行为与 ContentEncoder 一致")
    enc0 = SemanticEncoder()
    base = ContentEncoder()
    same = all(
        enc0.similarity(enc0.features(t), enc0.features("它是什么颜色的狗"))
        == base.similarity(base.features(t), base.features("它是什么颜色的狗"))
        for t in ("它是什么狗", "那狗是啥色", "1+1"))
    print(f"  未训练时 features/similarity 与 ContentEncoder 完全一致：{same}")
    print(f"  可直接注入 GeneralMemory：{enc0}")

    line("[2] 用 KV 映射训练语义网络（同义问法拉近、异义推远）")
    weights = train_semantic(pairs, steps=600, hidden=64, out_dim=64,
                             log=print)
    tmpdir = tempfile.mkdtemp()
    weights_path = os.path.join(tmpdir, "semantic.npz")
    from ultra_instinct.training.semantic_train import save_weights
    save_weights(weights, weights_path)
    enc = SemanticEncoder.from_weights(weights_path)
    print(f"  训练完成：{enc}")

    line("[3] 练一条问法，问它的同义变体（语义命中）")
    mem = GeneralMemory(enc, min_episodes=2, gist_threshold=0.4)
    drill(mem, "它是什么颜色的狗", "它是一条黑色的德牧", 2, 0)
    for q in ("它是什么颜色的狗",   # 练过的原话
              "它是什么狗",         # 同义、没逐字练过
              "这狗什么颜色",       # 同义、没逐字练过
              "那狗是啥色"):        # 同义、没逐字练过
        a = mem.query(q)
        print(f"  {q:<14} -> {str(a.response):<12} "
              f"来源={a.source:<9} 相似度信任={a.weight:.0%}")

    line("[4] 精确路径仍在：一字不差是 verbatim、微秒")
    a = mem.query("它是什么颜色的狗")
    print(f"  原话来源={a.source}（verbatim 恒满信，不经过网络判定语义）")
    print(f"  语义网络只负责把「没练过但有字面桥梁」的问法接上，"
          "不牺牲已练熟内容的精确命中。")


if __name__ == "__main__":
    main()
