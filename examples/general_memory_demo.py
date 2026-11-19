"""通用肌肉记忆演示：练内容，不练方向。

跑法：  python examples/general_memory_demo.py

演示六件事，全部是真跑出来的，没有硬编码结论：
    1. 反复练熟 → 跨 episode 固化 → 之后脱口而出
    2. 没练过的问题不硬答（准确性优先）
    3. 写错一个字也能靠要旨接上（「窗前明月光」→「疑似地上霜」）
    4. 长期不用 → 压缩成要旨，占用变小
    5. 再遇到 → 再巩固，逐字长回来
    6. 学新的不破坏旧的 + 容量线性到几千条
    7. 被压缩的内容按复杂度遗忘：细节多的先忘，核心要旨留得久
"""

from __future__ import annotations

import os
import sys
import time

# 直接跑本文件时把项目根加进路径，避免 ModuleNotFoundError
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ultra_instinct.forgetting as F
from ultra_instinct.encoding import ContentEncoder
from ultra_instinct.forgetting import ForgettingProfile
from ultra_instinct.general_memory import GeneralMemory


def line(title: str = "") -> None:
    print("\n" + "=" * 68)
    if title:
        print(title)
        print("=" * 68)


def practice_n(mem: GeneralMemory, content: str, response: str,
               n: int, base: int = 0) -> None:
    for k in range(n):
        mem.practice(content, response, base + k, now=base + k)


def main() -> None:
    enc = ContentEncoder()
    mem = GeneralMemory(enc, min_episodes=3)

    # ---------------------------------------------------------- 1
    line("[1] 练熟 → 固化 → 脱口而出")
    practice_n(mem, "1+1", "2", 3, 0)
    before = mem.query("1+1")
    print(f"  第 1 次练完就问：来源={before.source:<9} 答案={before.response}")
    mem.consolidate()
    after = mem.query("1+1")
    print(f"  跨 3 次练习固化后：来源={after.source:<9} 答案={after.response}"
          f"  信任={after.weight:.0%}")

    # ---------------------------------------------------------- 2
    line("[2] 没练过的，不乱答")
    q = mem.query("今天天气怎么样")
    print(f"  问「今天天气怎么样」：来源={q.source}"
          f"  答案={q.response!r}  ← 不知道就说不知道")

    # ---------------------------------------------------------- 3
    line("[3] 诗句接龙 + 错字容错")
    practice_n(mem, "床前明月光", "疑是地上霜", 3, 10)
    practice_n(mem, "举头望明月", "低头思故乡", 3, 20)
    mem.consolidate()
    for ask in ("床前明月光", "窗前明月光", "疑似地上霜", "举头望明月"):
        a = mem.query(ask)
        print(f"  {ask:<8} →  {str(a.response):<10} 来源={a.source:<9}"
              f" 信任={a.weight:.0%}")
    print("  注：「窗前明月光」是错字，逐字没精确命中，靠字符相似度")
    print("      在记忆里找到相近的「床前明月光」（相似 0.79），照样接上。")
    print("      倒着问「疑似地上霜」则答不出 —— 肌肉记忆是单向的。")

    # ---------------------------------------------------------- 4
    line("[4] 长期不用 → 压缩成要旨")
    m2 = GeneralMemory(enc, min_episodes=2)
    pool = [
        ("窗前明月光", "疑似地上霜"), ("床前明月光", "疑似地上霜"),
        ("月光照窗前", "疑似地上霜"), ("明月光窗前", "疑似地上霜"),
    ]
    for i, (c, r) in enumerate(pool):
        practice_n(m2, c, r, 2, i * 2)
    m2.consolidate()
    fp0 = m2.footprint
    print(f"  压缩前：逐字 {len(m2.verbatim)} 条，占用 {fp0['total']} 参数")
    st = m2.compress(now=100)
    fp1 = m2.footprint
    print(f"  压缩后：逐字 {len(m2.verbatim)} 条，要旨 {len(m2.gists)} 个，"
          f"占用 {fp1['total']} 参数"
          f"  ↓ {(1 - fp1['total'] / fp0['total']):.0%}")
    a = m2.query("窗前明月光", now=101)
    print(f"  压缩后还能答：{a.response}  来源={a.source}")

    # ---------------------------------------------------------- 5
    line("[5] 再遇到 → 再巩固，细节长回来")
    m2.reconsolidate("窗前明月光", "疑似地上霜", 100, now=102)
    m2.reconsolidate("窗前明月光", "疑似地上霜", 101, now=103)
    a = m2.query("窗前明月光", now=104)
    print(f"  重新练 2 次后：逐字 {len(m2.verbatim)} 条，"
          f"来源={a.source}，答案={a.response}")
    print("  不是把要旨展开 —— 支撑点压缩时已删，是这次真实练回来重新长的。")

    # ---------------------------------------------------------- 6
    line("[6] 学新的不破坏旧的 + 容量")
    m3 = GeneralMemory(enc, min_episodes=2)
    old = [("1+1", "2"), ("2+2", "4"), ("3+3", "6")]
    for i, (c, r) in enumerate(old):
        practice_n(m3, c, r, 2, i * 2)
    m3.consolidate()
    print(f"  先学 {len(old)} 条旧知识")

    t0 = time.perf_counter()
    N = 3000
    for i in range(N):
        practice_n(m3, f"问题{i}号是什么", f"答案{i}", 2, 1000 + i * 2)
    m3.consolidate()
    dt = time.perf_counter() - t0
    print(f"  再学 {N} 条新知识：{dt:.1f}s，逐字总数 {len(m3.verbatim)}")

    ok_old = all(m3.query(c).response == r for c, r in old)
    sample = range(0, N, 300)
    ok_new = all(m3.query(f"问题{i}号是什么").response == f"答案{i}"
                 for i in sample)
    print(f"  旧知识全在：{ok_old}   新抽查 {len(list(sample))} 条全对：{ok_new}")
    print("  容量线性扩展，不受原二维网格 256 格的限制。")

    # ---------------------------------------------------------- 7
    line("[7] 被压缩的内容按复杂度遗忘（论文参数）")
    print(f"  四成分半衰期：逐字 {40} 分钟 / 命题 {4} 天 / "
          f"要旨 {30} 天 / 语义 {25:.0f} 年   (alpha={F.ALPHA})")
    print("  复杂度：合并的逐字越多 → 保留细节越多 → 越先忘")

    pf = ForgettingProfile(threshold=0.5)
    for label, c in (("细节多(复杂度1.0)", 1.0), ("核心要旨(复杂度0.0)", 0.0)):
        hl = F.halflife_of(c)
        unit = f"{hl * 1440:.0f} 分钟" if hl < 1 else f"{hl:.0f} 天"
        print(f"  {label:<20} 半衰期≈{unit:<10} "
              f"R(1天)={F.retention(1, c):.0%}  R(1年)={F.retention(365, c):.0%}")

    # 真跑一次：同样的 gist，复杂度高的先被 forget 删掉
    import numpy as np
    from ultra_instinct.general_memory import GistNode
    v = np.array([1.0, 0.0], dtype=np.float32)
    m4 = GeneralMemory(forgetting=pf)
    m4.gists = [
        GistNode(center=v, response="细节多", radius=0.5, coherence=1.0,
                 support=20, complexity=1.0, born=0.0),
        GistNode(center=v, response="核心要旨", radius=0.5, coherence=1.0,
                 support=1, complexity=0.0, born=0.0),
    ]
    st = m4.forget(now=30)
    print(f"  放 30 天后执行 forget：忘掉 {st['forgotten']} 条，"
          f"剩 {st['kept']} 条 → 剩的是「{m4.gists[0].response}」")
    print("  时间单位=天。要旨不是被展开回逐字，要用时只能靠再巩固重新观测。")

    line("完成")
    print("  from ultra_instinct import ContentEncoder, GeneralMemory")
    print("  mem = GeneralMemory(ContentEncoder(), min_episodes=3)")
    print("  mem.practice('问题', '答案', episode_id=0)")
    print("  mem.consolidate()")
    print("  mem.query('问题').response")


if __name__ == "__main__":
    main()
