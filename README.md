# Ultra Instinct

纯软件的「肌肉记忆」系统：练熟的内容直接查表调用，没练过的不硬答（控制版走网络现算，通用记忆版回落浅记忆或拒答），学新内容不覆盖旧记忆。推理路径不依赖深度学习框架，普通 CPU 即可运行。

[English version](README_EN.md)

## 前言

在现今各种大模型的推出，较全面的模型越来越来多。我只想做一个小而精的模型基座，这个系统的命名来自于龙珠：超中孙悟空的“自在极意功”形态。

它模拟了人类大脑的肌肉记忆系统，一个参数得到反复训练后会从短期记忆到长期记忆，长期记忆中不常用的又会被简化为主旨，再得到同样的行为时候又会重新丰富细节。如果超长时间不用也就进行删除，这个跨度符合人类的记忆为25年。

我做这个本意只是想在大模型众多的环境中做一个基础细分模型，它的应用性还没确定，但我觉得它可以作为通用大模型的前置，结合人类大脑的工作机制提供快速的肌肉潜意识反应。以后可能会在机器人领域有所应用。

在默认字符编码器下，练熟的内容走的是直接查表：内核实测逐字命中的中位延迟约 18 微秒（p99 约 36 微秒），要旨模糊命中约 35 微秒（按当前内核 `decay(now=30)` 沉淀后复测 p50 仍为 35.5 微秒）；即便经过完整的 HTTP 服务，逐字查询端到端中位数也仅约 1.3 毫秒，流式输出的首个字块约 25 毫秒。注入本地语义编码器后，短文本编码 p50 约 60 微秒；约 200 字长文本的特征提取与一次查询分别约 987 微秒与 1008 微秒，HTTP 端到端 p50 约 1.6~2.8 毫秒，仍是毫秒级。作为参照，人类对简单刺激的有意识反应通常在十分之几秒量级——也就是说，对已经练熟的内容，它的反应比人还快上一个数量级，几乎不需要“想”，这正是肌肉记忆要的效果。

它的长处在于快、省、稳、准：练熟的内容查表即得，不依赖深度学习框架，普通 CPU 就能跑；默认特征用字符 n-gram 现算、不需要词表或嵌入模型，需要「说法不同、答案相同」时可注入本地自写的小语义网络；学习新内容时与旧记忆冲突就分离存放，不整体覆盖；把「想起一个候选」和「够格说出来」分成两级，练过的才授权输出，没练过的宁可拒答也不编。记忆还能落盘保存、跨进程恢复。

它的局限同样来自这套机制本身：这是「背下来后脱口而出」，不是推理——背过 `1+1=2` 能答，没背过的 `137×842` 答不出，现算需要另接规则引擎；要旨联想是单向的，顺着能接、反着问不出；默认相似判定基于字面 n-gram，换个说法可能匹配不上，可选语义网络只学训练数据里出现过的等价问法，不是通用语义理解；目前是离线练习→固化→只读，没有在线学习；记忆集中在一个进程里，多实例共享同一份文件需要自己加锁。

## 一、系统构思

核心命题：**练熟到不用想**。这与装进去的是动作方向还是文字无关——大脑把重复练习固化为快速检索，本系统用同一套机制模拟这件事。

由此得到三条设计目标：

1. **熟能生巧**：反复练习且结果一致的内容，提升为可直接查表的记忆，用时 O(1) 取出。
2. **不会就不硬答**：控制版走通用网络现算；通用记忆版回落浅记忆试探或拒答，不给编造的答案。
3. **学新不覆旧**：新内容与旧记忆冲突时分离存储，不整体覆盖；长期不用的压缩成要旨，需要时靠重新观测补回细节。

系统有两套内核，共用同一套分级记忆逻辑，只是键的形式不同：

| 内核                                            | 输入 → 输出               | 适用场景            |
| --------------------------------------------- | --------------------- | --------------- |
| **通用内容**（`encoding.py` + `general_memory.py`） | 任意文本/字节 → 任意文本        | 事实、话术、诗句接龙、背诵问答 |
| **二维控制**（`state.py` + `policy.py`）            | `(Δx, Δy, 技能)` → 方向向量 | 机械臂到达、低维运动控制    |

对外有三个入口，按需选用：`GeneralMemory`（通用记忆 Python 接口）、`Policy`（控制版 Python 接口）、`api.py`（OpenAI 兼容的 HTTP 流式接口，底层走通用记忆）。

## 二、设计思路

### 2.1 分级记忆：浅 → 逐字 → 要旨

记忆分三级，加一条回填路径，模拟「练几次进浅记忆、够次数进深记忆、长期不用压成要旨、再遇到补回细节」：

```
练习 → 浅记忆（一次就写，容量小，衰减快）
        ↓ 跨多个 episode 重复且答案一致
      逐字记忆（细节完整，精确查表）
        ↓ 长期不用 + 块内动作/答案一致
      要旨（只留核心 + 局部斜率，体积大幅缩小）
        ↑
        └── 再次用到 → 在命中小块上重新长出逐字（再巩固）
```

四条关键取舍：

- **要旨保留斜率，不只留均值**。压缩的工程含义是丢掉二阶曲率、保留一阶斜率，而不是把一堆细节平均掉。
- **矛盾块拒绝压缩**（pattern separation）。相似输入若给出不同答案，合并会平均出谁都不对的结果，必须分离。
- **补细节靠重新观测，不是解压**。逐字痕迹和要旨平行存储，压缩时支撑点已删除，无法从要旨还原；再巩固只能用新样本在局部重新生长。
- **遗忘是功能**。压缩内容按复杂度逐步遗忘：细节多的先忘、核心要旨留得久，符合「泛化优于精确存储」。

### 2.2 快慢双通路

这一节讲的是**二维控制内核**的读路径；通用内容记忆不走网络混合，而是 4.1 的三级回落（逐字 → 要旨 → 浅记忆 → 拒答）。

- **写路径（离线，允许慢）**：练习 → 候选池 → 跨 episode 达标 → 一致性检验 → 入表。
- **读路径（在线，要快）**：输入 → 查表得权重 w → w 高直接出表值，w 低走网络，中间混合。

慢通路负责学和新奇情况泛化，快通路负责熟练情况零决策。两者互补：只有快通路覆盖不足时无法泛化，只有慢通路则每次都要现算。

混合不是简单二选一，而是按固化权重 w 连续插值：`混合 = w × 表值 + (1−w) × 网络值`。这是整个项目最关键的一条设计依据——实测在固化区边界，硬切换的两路指令夹角平均 **96.2°**、最大 103.7°，100% 超过 30°；连续混合后平均 **6.0°**、最大 10.3°，0% 超 30°。

还有两个容易踩的坑：

- **权重必须来自固化掩码的插值，不能用「到最近格中心的距离」**。距离本身在格边界不连续，拿它当权重等于没平滑，实测跳变仍是 90°。
- **两路接近反向时要防退化**。夹角趋近 180° 时混合向量模长趋零，归一化会把数值噪声放大成随机方向；`blend()` 设了 `degenerate_eps`（默认 0.2），模长低于它就不归一化。

### 2.3 准确性优先：两级判定

把「想起一个候选」和「够格说出来」分成两级，避免模糊联想被当成确定答案：

- **联想激活**：由相似度阈值决定，只判断有没有候选。
- **输出授权**：由置信度门 `min_confidence` 决定。练熟的逐字记忆恒授权；要旨/浅命中要 `weight >= 门` 才授权。
- 非严格模式返回答案并附 `confident` 标记；严格模式（`strict=true`）下低置信一律返回空，供执行类场景使用。

### 2.4 关键设计决策

下表主要来自二维控制内核的取舍；通用内容记忆沿用跨 episode 固化、numpy 推理等原则，但不做双线性插值与转角限幅。

| 决策                      | 原因                      |
| ----------------------- | ----------------------- |
| 状态用相对量 `(Δ, 技能)`，不用绝对坐标 | 所有目标共享一张表，相对量可泛化        |
| 状态必须带技能标识               | 同一位置不同意图动作相反，缺这维必然冲突    |
| 查表用双线性插值，不取最近格          | 低分辨率 + 插值优于高分辨率最近邻，内存更小 |
| 混合权重来自固化掩码的插值           | 保证边界连续，避免硬切换跳变          |
| 固化跨 episode 计数          | 同一 episode 重复不算多次验证     |
| 可靠性靠失败归因，不靠事前门槛         | 覆盖率与准确率之间取更划算的取舍        |
| 推理用 numpy 不用 torch      | 去掉框架调度开销，单步快数倍          |

## 三、技术架构

### 3.1 分层

| 层       | 文件                                                 | 职责                                    |
| ------- | -------------------------------------------------- | ------------------------------------- |
| 状态/内容编码 | `state.py`、`encoding.py`、`semantic.py`             | 观测 → `(Δ, 技能)` 或 内容 → 键 + 特征（可注入语义编码） |
| 量化与查表   | `quantize.py`、`memory.py`                          | 坐标 → 格索引 + 插值权重；逐字表查询/撤销/持久化          |
| 分级记忆    | `hierarchy.py`、`general_memory.py`、`forgetting.py` | 三级记忆、压缩、再巩固、按复杂度遗忘                    |
| 慢通路     | `net.py`、`training/`                               | numpy MLP 前向；torch 离线训练并导出 npz        |
| 安全与路由   | `blend.py`、`safety.py`、`policy.py`                 | 加权混合、转角限幅、失败归因                        |
| 服务      | `api.py`                                           | OpenAI 兼容 HTTP 流式接口                   |

### 3.2 目录

```
UltraInstinct/
├── ultra_instinct/
│   ├── state.py / encoding.py      状态与字符内容编码
│   ├── semantic.py                 可注入语义编码器（本地自写网络）
│   ├── quantize.py / memory.py     网格量化与逐字表
│   ├── hierarchy.py                分级记忆（控制版）
│   ├── general_memory.py           通用内容记忆
│   ├── forgetting.py               分层遗忘（论文参数）
│   ├── net.py / blend.py / safety.py  慢通路、混合、限幅
│   ├── policy.py                   控制版对外接口
│   ├── api.py                      HTTP 流式服务
│   ├── training/                   离线训练（torch，含 semantic_train.py）
│   └── consolidation/              固化流水线
├── examples/                       五个演示 + KV/长文本训练样例（见 4.1.1、5.3）
├── checkpoints/                    语义权重与记忆文件（语义权重与记忆须配套使用）
├── benchmarks/                     9 个基准脚本 + 实测报告（见 3.3）
├── tests/                          176 个单元测试（含语义、API 门槛、渐进退化）
├── configs/default.json            配置样例（当前代码未加载，默认值见第六章）
└── requirements.txt                依赖
```

### 3.3 实测指标

`benchmarks/bench_metrics.py` 跑端到端实测，报告落在 `benchmarks/metrics_report.json`。先起服务再跑脚本：

```bash
python -m ultra_instinct.api --host 127.0.0.1 --port 8000
python benchmarks/bench_metrics.py
```

一次典型的实测结果（训练数据为 18 条简单内容，练出 6 条逐字；内核微基准用默认 `ContentEncoder`，要旨由 `decay(now=30)` 按当前默认渐进退化沉淀）：

| 指标               | 实测值                                    |
| ---------------- | -------------------------------------- |
| 练过的命中率           | 100%（6/6）                              |
| 没练过的拒答率          | 100%（3/3）                              |
| 错字容错             | 「窗前明月」→「疑似地上霜」（浅记忆，weight 0.5）         |
| 逐字查表延迟 p50 / p99 | 18.1µs / 36.4µs                        |
| 要旨查表延迟 p50 / p99 | 35.5µs / 90.5µs                        |
| 未命中回落 p50        | 137.3µs                                |
| HTTP 逐字端到端 p50   | 1.32ms                                 |
| SSE 流式首块 / 总耗时   | 25.1ms / 25.7ms（7 块，正确拼回答案并收 `[DONE]`） |
| 记忆占用             | 逐字 1542 + 浅记忆 4644 = 6186 参数           |
| 要旨半衰期（复杂度 高/中/低） | 0.03 天 / 1.07 天 / 50.37 天              |

这些数字是一次真实运行的记录，随机器与数据而变，用于量级参考而非承诺。

语义/长文本的补充实测（另一组数据，不与上表混算）：

| 指标                              | 实测值                                                          |
| ------------------------------- | ------------------------------------------------------------ |
| 同义组相似度（训练前 → 训练后）               | 0.154 → 0.996                                                |
| 异义组相似度（训练后）                     | -0.463                                                       |
| 语义短文本编码 p50                     | 60µs（默认字符编码 58µs）                                            |
| 约 200 字长文本特征提取 / 查询             | 约 987µs / 1008µs                                             |
| 语义服务 HTTP 端到端 p50               | 1.6~2.8ms                                                    |
| 门槛 0.85 的长文本命中 / 域外拒答（非 strict） | 同义问法 8/8 返回同一答案，域外输入 4/4 返回空                                 |
| 短问法的授权状态                        | `source=episodic`、`weight=0.5`、`confident=false`（strict 会挡住） |

`benchmarks/` 下另有八个专项脚本，各自回答一个设计问题，结果写在同目录同名 json 中：

| 脚本                   | 回答的问题                        |
| -------------------- | ---------------------------- |
| `bench_continual.py` | 学新内容是否覆盖旧记忆（学新不覆旧）           |
| `bench_trust.py`     | 固化区边界上硬切换 vs 连续混合的跳变角        |
| `bench_hierarchy.py` | 分级记忆相对单层的覆盖率 / 占用收益          |
| `bench_safety.py`    | 量化格数、转角限幅对动作误差的影响            |
| `bench_scaling.py`   | 容量增长时线性查表 vs KNN 的延迟曲线       |
| `bench_torch.py`     | numpy 前向 vs torch 的单步 / 批量延迟 |
| `bench_capacity.py`  | 不同网络规模的容量与延迟                 |
| `bench_latency.py`   | numpy 前向的基线延迟分解              |

## 四、核心逻辑说明

### 4.1 通用内容：`GeneralMemory`

**编码**（`encoding.py`）：精确键用 `blake2b(归一化文本)` 取 16 字节，跨进程稳定；特征向量用字符 1/2/3-gram + hashing trick 得 256 维 L2 归一向量，用于模糊匹配，相似度就是归一化向量的点积（余弦）。用字符 n-gram 而非词表，是为了对中文友好（「窗前明月光」和「床前明月光」只差一字、向量仍接近）、无需预先固定词表，且维度不随内容增长。

这里用 `blake2b` 而不是 Python 内置 `hash()` 是必须的：`hash()` 带随机盐，每次进程启动都变，存进检查点的键下次启动就对不上了。归一化用 NFKC + 去空白 + 小写，抹掉全角半角等无意义差异，但不碰实质文字。

**读路径**（`query()`，三级依次回落）：

1. 逐字：精确键命中 → 直接返回，`weight=1.0`，恒授权。
2. 要旨：与要旨中心相似度超过覆盖半径 → 返回要旨答案，`weight = 相似度 × coherence × 保持率`，过门才授权。
3. 浅记忆：相似情景表决 → 返回表决结果，`weight` 上限 0.5（在 `GeneralMemory.query()` 内封顶，使未固化内容默认过不了 0.6 的门）。
4. 都没命中 → 返回 `None`，不编答案。

**命中不产生训练**：`query()` 命中只刷新 `hits` / `last_used`（保鲜，见 4.1.2），不写 `_pending`、不计入跨 episode 的巩固次数——调用一次**不会**增加「短期 → 长期」的固化进度，也不会把命中的问法当成新训练样本。推进度只能靠 `practice()` / `reconsolidate()`；走 HTTP `/v1/chat/completions` 的查询永远不推进度。这样设计是为了不把试探性命中当成真值给自己投票，把错答案越攒越牢。

**写路径**：`practice()` 写入浅记忆（同一 episode 多次只计一次）→ `consolidate()` 跨 episode 达标且多数答案一致才提升为逐字 → `compress()` 把相似逐字聚簇、答案一致才合并成要旨并真删逐字（默认渐进退化下只处理 `detail ≤ detail_threshold` 的条目，见 4.1.2）→ `reconsolidate()` 在要旨命中处重新观测、重新长出逐字。

`compress()` 的一致性判定是**簇内多数投票**：`coherence = 最多答案的票数 / 簇大小`，低于 `coherence_min` 就拒绝合并（相似输入给出不同答案，合并只会平均出谁都不对的结果，必须分离）。要旨的覆盖半径取簇内各点到中心的**最小相似度**，保证整个簇都被覆盖；中心是簇内特征的均值再归一化，复杂度按 support（簇大小）用 `complexity_scale` 归一化，交给遗忘层决定半衰期。

#### 4.1.1 语义编码器（`semantic.py`，可选注入）

默认 `ContentEncoder` 用字符 n-gram，只认字面重叠——「它是什么狗」和「那狗是啥色」字面差很远，命中不了。若部署方希望**说法不同、答案相同**，可注入 `SemanticEncoder`：一个本地自写的小网络，把字符 n-gram 向量映射成「意思向量」，同义问法的向量被拉近。

**训练**：把部署方的 KV 映射写成 `{答案: [同义问法, ...]}`（见 `examples/kv_example.json`），用监督对比损失训练——同一答案的问法互为正样本、不同答案互为负样本：

```bash
python -m ultra_instinct.training.semantic_train \
    --pairs examples/kv_example.json --out checkpoints/semantic.npz
# 长文本等价类（同一答案配一篇长文 + 两种短问法）：
python -m ultra_instinct.training.semantic_train \
    --pairs examples/long_text_kv.json --out checkpoints/semantic_longtext.npz
```

`examples/long_texts.json` 是同一批长文本的原文清单，便于阅读与改语料；`examples/long_text_kv.json` 才是训练输入。

训练用 torch，导出成 `net.py` 的 MLP 格式 npz；推理端只有 numpy 前向，零 torch 依赖。实测同义组相似度从 0.154 升到 0.996、异义组被推到 -0.463；**编码延迟只比纯字符多约 2µs**（p50 58→60µs），因为瓶颈在 n-gram 哈希而非矩阵乘。

**使用**：未训练时 `SemanticEncoder.features()` 退化为 `ContentEncoder` 的字符特征，不训练也能用；注意此时 `dim` 默认为 64（占位，与 256 维字符特征不等），加载权重后以权重的输出维度为准。加载权重后直接注入，内核无需改：

```python
from ultra_instinct import GeneralMemory, SemanticEncoder
enc = SemanticEncoder.from_weights("checkpoints/semantic.npz")
mem = GeneralMemory(enc)
mem.practice("它是什么颜色的狗", "它是一条黑色的德牧", 0)
mem.consolidate()
mem.query("那狗是啥色").response   # 换说法，语义命中同一答案
```

精确路径保持不变：一字不差的输入仍走 `blake2b` 逐字命中、微秒级；语义网络只负责把训练中归为同一答案的换说法接上。

注意语义命中不等于授权命中：没练过的换说法可能只到浅记忆（`weight=0.5`、`confident=false`），非 strict 返回同一答案供参考，strict 下会被授权门挡住；要让换说法也能授权执行，需把它也练到逐字。

**能力边界**：网络只学训练数据里出现过的等价关系，不是通用语义理解。「客户编号」和「用户ID」若没在同一答案的训练组里出现过，零重叠就无法拉近；要覆盖就得把它们作为同义问法写进 KV 映射重训。这是「轻量、自写、微秒」换来的必然边界——通用语义需要预训练大模型，那不是本项目的取舍。

**持久化**：`save()` 不存网络权重；编码器只存 `dim / ngrams / normalize` 三项配置，完整记忆状态另存于同一文件。恢复时仍须显式注入同一语义编码器——`GeneralMemory.load(path, encoder=enc)`，否则会退回默认 `ContentEncoder` 导致特征维度不匹配。

#### 4.1.2 渐进退化：从长期记忆到要旨的过渡（`gradual_decay`，默认开启）

**要解决的问题**：早期实现里「逐字 → 要旨」靠手动调用一次 `compress()`，瞬间把细节删光——这是开关，不是过程。人脑是练熟后先以**完整细节**长期保存，经过一段时间不用，细节才渐渐褪去、简化成要旨；期间再用到就重新保鲜。因此这套时间驱动的退化**默认开启**，无需显式打开。

开启后（`gradual_decay=True`，默认）：

- 每条逐字带一个**细节完整度** `detail = (1 + age/τ)^(-α)`，`age = now − last_used`。刚固化或刚用过时 = 1.0（细节完整），随时间连续下降。
- **用进保鲜**：`query()` 命中、`reconsolidate()` 再巩固都会刷新 `last_used`，细节回满——常用的记得清楚，不用的才模糊。
- **分次沉淀**：`decay(now)`（或 `compress()`）只把 `detail ≤ detail_threshold` 的逐字合并成要旨，细节仍高的原样留在逐字层。于是同一批记忆会随时间**先后**退化，而非一次抹平。
- **再巩固回满**：重新观测让褪色条目长回完整细节（0.09 → 1.0），不是从要旨解压。
- 时间锚点用真实 `now`（`born/last_used` 在固化时写入），随记忆一起落盘；旧的"born 存序号"隐性 bug 一并修正。

**关闭**（`gradual_decay=False`）：退回旧语义——`detail()` 恒为 1.0、`decay()` 是空操作、`compress()` 即时一步删。提供此开关只为兼容旧调用方，默认不建议关闭。从旧版记忆文件恢复（配置里没有该字段）时也按**开启**处理。

```python
from ultra_instinct import GeneralMemory
m = GeneralMemory(gradual_decay=True, detail_threshold=0.5, tau=3.0, alpha=1.0)
# ... practice / consolidate ...
m.decay(now=30)     # 时间推进到"第30天"，只沉淀已褪色的逐字
                    # 期间被 query 命中的条目 last_used 刷新，不会被沉淀
# 需要旧式"一步删光"时才显式关：GeneralMemory(gradual_decay=False)
```

### 4.2 遗忘（`forgetting.py`）

保持函数 `R(t) = (1 + t/τ)^(-α)`，`α=0.13` 由 Ebbinghaus（1885）原始 7 个数据点网格拟合（20min 58.2% / 1h 44.2% / 8.8h 35.8% / 1d 33.7% / 2d 27.8% / 6d 25.4% / 31d 21.1%），对全部点最大误差 < 0.03。

**为什么是幂律不是指数**：指数衰减隐含一个特征时间常数，实测对不上；Wixted & Ebbesen (1991)、Anderson & Schooler (1991) 都支持幂律。要注意这组数据来自单被试、无意义音节、再学习法，它是**曲线形状的基准**，不是普适常数。

记忆强度是四个幂律成分的加权和，复杂度决定权重——细节越多，短寿命成分权重越大，忘得越快。**四层不是四条不同的曲线**，而是保持 α 不变、把同一条 Ebbinghaus 曲线横向拉伸（只改 τ，`R_layer(t) = R_eb(t×s)`）：

| 成分            | 半衰期   | 来源              |
| ------------- | ----- | --------------- |
| 逐字/表层细节       | 40 分钟 | Kintsch 1990    |
| 命题/textbase   | 4 天   | Kintsch 1990    |
| 核心要旨          | 30 天  | Sacripante 2023 |
| 语义/permastore | 25 年  | Bahrick 1984    |

复杂度到权重是**线性映射、没有额外折衷参数**（`c, (1−c)/3, (1−c)/3, (1−c)/3`），这样「复杂度高的先忘」在任意 `age>0` 都单调成立，可被单元测试直接验证。机制来自 RAFT 模型（Fisher & Radvansky 2019）：一条 gist 若保留大量逐字细节，短寿命成分权重大就先掉；高度压缩只剩核心要旨的，长寿命成分权重大就抗遗忘。

**练习只改截距、不改斜率**（Slamecka & McElree 1983；Rivera-Lares 2022）——练习次数作用于 strength 的初值，不改变上面的 α / τ，所以「多练」提高起点而非减缓遗忘速度。

`forget(now)` 把保持率低于阈值的要旨真删除（默认关闭，需注入 `ForgettingProfile` 才启用）。时间单位是**天**。

### 4.3 二维控制：`Policy`

`act()` 一次决策：状态编码 → 查表得权重 → 高权重出表值 / 低权重走网络 / 中间混合 → 转角限幅 → 返回结果（`action, weight, source, net_called, clamped`）。`begin_episode()` / `end_episode()` 圈定一个任务周期，失败时撤销本次靠手感做的格子。

**失败归因为什么放在 episode 级，不是单步级**：单步判据试过两版都失败——「距离没变小就算失败」把 99.7% 的决策判死，因为绕行时距离本来就持平；「比最好值差 5% 就算退步」把 118 条正确记忆误杀 52 条。根因是偏差 90° 时距离恒定，任何距离类过程指标都测不出「方向偏了」。改用「这次任务整体成不成功」这个确定信号反推哪几步手感可疑，粒度粗但不会误杀，而且归因方向正确：是手感错了才导致失败。

顺带说明，`RateLimiter` 的单步限幅是兜底不是替代混合——混合后残余跳变最大 10.3°（slerp 后降到 `≤8.0°`）；且它是有状态的，换任务必须 `reset()`，否则上一段轨迹的方向会拖住新任务第一个动作。

**训练一条控制技能**（torch 离线训练，导出后推理零 torch 依赖）：`examples/reach_demo.py` 是完整可跑的四步流水，产出 `checkpoints/reach`。

1. **练慢通路**：`TorchMLP` + `TrainConfig` + `train_skill()`，用 `ReplayBuffer` 联合回放防止学新技能忘旧技能；`export_mlp()` 导出成 `net.py` 的 numpy MLP。
2. **收成功经验**：跑若干 episode，把成功的 `(delta, action)` 喂给 `CandidatePool.record()`，按格聚合、跨 episode 计数。
3. **固化成表**：`pool.promoted()` 取跨 `min_episodes`（默认 5）个 episode 成功且动作离散度（`max_spread`，默认 0.25）达标的格子，`compile_memory()` 编译成 `LUTMemory`。
4. **组装落盘**：`Policy(net=net, memory=memory, config=...)`，用 `save()` 写出 `config.json` / `net.npz` / `memory.npz` 三件套。

```python
import numpy as np
from ultra_instinct.training import TorchMLP, TrainConfig, ReplayBuffer, train_skill, export_mlp
from ultra_instinct.consolidation import CandidatePool, compile_memory
from ultra_instinct.quantize import Grid
from ultra_instinct import Policy, PolicyConfig

rng = np.random.default_rng(0)
net_t = TorchMLP(input_dim=3, hidden=64, output_dim=2)  # 2 维相对目标 + 1 维技能码
cfg = TrainConfig(steps=1500, lr=2e-3, batch=256)
train_skill(net_t, task_fn, cfg, rng, skill_id=0, n_skills=1, replay=ReplayBuffer())
net = export_mlp(net_t)

pool = CandidatePool(Grid(32, 1.0), n_skills=1, min_episodes=5)
# ... 跑 episode：pool.record(delta, skill_id=0, action, episode_id)
memory = compile_memory(pool.promoted(), Grid(32, 1.0), n_skills=1, act_dim=2)
Policy(net=net, memory=memory, config=PolicyConfig()).save("checkpoints/reach")
```

`task_fn` 由场景提供（`reach_demo.py` 里是二维到达的理想向量场）。技能标识必须拼进网络输入且在同一批里有变化，否则网络会忽略这一维（见 `training/trainer.py` 模块说明；推理端由 `encode_state` 自动拼接）。

### 4.4 持久化（`save` / `load`）

`GeneralMemory.save(path)` 把**完整状态**写成一个 JSON 文件（bytes 键与向量用 base64 编码）：三级记忆、巩固计数 `_pending`、固化仍需要的原文/特征缓存、全部构造参数（含 `gradual_decay` / `detail_threshold`）、逐字条目的 `born/last_used` 与 pending 的时间锚点，**以及 encoder 配置（dim / ngrams / normalize）**——key 与特征都由 encoder 生成，配置不存会导致恢复后维度错配崩溃或归一化不一致而静默查不到。原子写入（临时文件 + 替换），中途崩溃不会留下半个文件。`GeneralMemory.load(path)` 恢复后可原样继续练习——包括把没固化完的 pending 继续巩固；旧版本缺 encoder 或缺渐进退化字段的文件分别按默认 encoder、默认开启兼容加载（见 4.1.2）。

```python
mem.save("checkpoints/general_memory.json")
mem2 = GeneralMemory.load("checkpoints/general_memory.json")
# 用了非默认 encoder 也能正确恢复，无需重新传入：
# m = GeneralMemory(ContentEncoder(dim=64, normalize=False)); m.save(path)
# GeneralMemory.load(path).encoder.dim == 64 且 normalize == False
```

缓存只服务未固化条目的再巩固：每次 `consolidate` / `compress` / `reconsolidate` 后清掉已无引用的缓存项，不会随练习无限增长渗进文件。

HTTP 服务默认开启持久化：`python -m` 与 `uvicorn ultra_instinct.api:app` **两种启动方式都会恢复与落盘**（后者经 lifespan 初始化）。每次练习/压缩后落盘；查询命中统计按 `persist_query_every`（默认 50 次）节流落盘，停服时再 flush 一次——既不污染微秒级热路径，又保证正常关闭不丢统计。实测跨进程重启后记忆完整保留。

HTTP 目前**不暴露** `decay(now)`：`/v1/memories` 的 `compress` 以 `now=0` 调用，默认渐进退化下不会沉淀刚固化的逐字；需要按时间退化请用 Python 接口 `m.decay(now)`（见 4.1.2）。

## 五、如何使用

### 5.1 安装与运行

```bash
pip install -r requirements.txt

# 通用内容记忆演示（1+1、诗句、错字、压缩、再巩固、遗忘）
python examples/general_memory_demo.py
# 语义编码器演示（KV 映射训练 → 换说法命中同一答案）
python examples/semantic_demo.py
# 长文本语义训练（训练输入见 examples/long_text_kv.json）
python -m ultra_instinct.training.semantic_train \
  --pairs examples/long_text_kv.json --out checkpoints/semantic_longtext.npz
# 二维控制演示
python examples/reach_demo.py
# HTTP 客户端演示（需先按 5.3 启动服务）
python examples/api_client_demo.py
# 输出授权门验证：默认 vs strict 对比（需先启动服务）
python examples/confidence_gate_client.py
# 端到端实测（需先启动服务，结果写入 benchmarks/metrics_report.json）
python benchmarks/bench_metrics.py
# 单元测试
python -m unittest discover -s tests -t .
```

### 5.2 Python 接口

```python
from ultra_instinct import ContentEncoder, GeneralMemory
from ultra_instinct.forgetting import ForgettingProfile

mem = GeneralMemory(ContentEncoder(), min_episodes=3,
                    forgetting=ForgettingProfile())  # 不传则关闭遗忘
mem.practice("1+1", "2", episode_id=0)   # 练；换 episode_id 多练几次
mem.consolidate()                        # 巩固：跨 episode 达标的转逐字
mem.compress(now=10)                     # 可选：压成要旨（now 单位=天）
answer = mem.query("1+1").response       # '2'
mem.reconsolidate("1+1", "2", episode_id=3, now=11)  # 要旨命中后用新观测重新长出逐字
mem.forget(now=100)                      # 可选：删掉已遗忘的要旨
```

控制版：

```python
from ultra_instinct import Policy
policy = Policy.load("checkpoints/reach")
policy.begin_episode(skill_id=0)
out = policy.act(delta, skill_id=0)      # ActResult(action, weight, source, ...)
policy.end_episode(success, skill_id=0)
```

### 5.3 HTTP API

```bash
python -m ultra_instinct.api --host 127.0.0.1 --port 8000
# 可选：--min-confidence 0.6（输出授权门）、--no-forgetting（关闭遗忘）
#       --memory-file <路径>（默认 checkpoints/general_memory.json，存在则自动恢复）
#       --no-persist（关闭持久化，退回纯内存、重启即丢）
#       --semantic-weights <npz>（注入语义编码器，见 4.1.1）
#       --gist-threshold 0.55（相似判定门槛；语义场景建议 0.85，挡住域外误命中）
# uvicorn 直启时改设环境变量 UI_SEMANTIC_WEIGHTS / UI_GIST_THRESHOLD
```

用语义编码器启动（换说法也能命中同一答案；门槛调高避免无关输入误召回）：

```bash
python -m ultra_instinct.api --host 127.0.0.1 --port 8000 \
  --semantic-weights checkpoints/semantic_longtext.npz --gist-threshold 0.85 \
  --memory-file checkpoints/semantic_test_memory.json
# 注意：语义权重与记忆文件的特征空间绑定，二者须配套；换权重请用新记忆文件。
```

直接测长文本（按上面命令启动服务后）：

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"大脑怎么把短期记忆变成长期记忆"}]}'
# → "记忆会从短期巩固到长期，细节被剥离成要旨，需要时再重建"
# usage: source=episodic、weight=0.5、confident=false（短问法走浅记忆表决）

curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"股票今天涨不涨"}]}'
# → 空内容，usage.source=none（域外输入不构成命中，不编答案）
```

执行类场景要开 `"strict":true` 时，上述短问法会被授权门挡住；需先把这些问法也练成逐字记忆。

| 端点                     | 方法   | 作用                                                                 |
| ---------------------- | ---- | ------------------------------------------------------------------ |
| `/health`              | GET  | 健康检查                                                               |
| `/v1/models`           | GET  | 模型列表（id = `ultra-instinct`）                                        |
| `/v1/memories`         | POST | 练习；`consolidate` 巩固、`compress` 请求字段压缩（默认 false，以 `now=0` 调用，见 4.4） |
| `/v1/memories/stats`   | GET  | 记忆占用与计数                                                            |
| `/v1/chat/completions` | POST | 对话；`stream=true` 走 SSE 流式                                          |

练习（同一条要练满 `min_episodes` 次，换不同 `episode_id`）：

```bash
curl -s http://127.0.0.1:8000/v1/memories -H 'Content-Type: application/json' \
  -d '{"items":[
        {"content":"1+1","response":"2","episode_id":0,"now":0},
        {"content":"1+1","response":"2","episode_id":1,"now":1},
        {"content":"1+1","response":"2","episode_id":2,"now":2}
      ],"consolidate":true}'
```

查询（非流式 / 流式）：

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"1+1"}]}'

curl -N http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"窗前明月光"}],"stream":true}'
```

非流式返回标准 `chat.completion`，`usage` 扩展了 `source` / `weight` / `confident`。加 `"strict":true` 可让低置信命中返回空。官方 OpenAI SDK 把 `base_url` 指向 `http://127.0.0.1:8000/v1` 即可直接调用（详见 `examples/api_client_demo.py`）。

## 六、核心参数及设置位置

所有参数都有默认值，按需在**构造时**覆盖。时间单位均为**天**。

| 参数                                            | 默认值                               | 设置位置                                                                               | 作用                           |
| --------------------------------------------- | --------------------------------- | ---------------------------------------------------------------------------------- | ---------------------------- |
| `dim` / `ngrams` / `normalize`                | 256 / (1,2,3) / True              | `ContentEncoder(...)`（`encoding.py`）                                               | 特征维度、n-gram 长度、文本归一化         |
| `steps` / `hidden` / `out_dim` / `tau` / `lr` | 800 / 64 / 64 / 0.1 / 3e-3        | `semantic_train.py` CLI 或 `train_semantic(...)`（见 4.1.1）                           | 语义网络训练步数、隐层、输出维度、对比温度、学习率    |
| `batch` / `seed`                              | 0 / 0                             | `semantic_train.py` CLI 或 `train_semantic(...)`                                    | 批大小（0=全量）、随机种子（保证可复现）        |
| `min_episodes`                                | 3                                 | `GeneralMemory(...)`、`Hierarchy(...)`；HTTP `min_episodes`                          | 跨几个 episode 才固化为逐字           |
| `stability`                                   | 0.6                               | `GeneralMemory(...)`                                                               | 固化所需的多数答案占比                  |
| `capacity`                                    | 64                                | `GeneralMemory(...)`、`EpisodicBuffer(...)`                                         | 浅记忆容量上限                      |
| `gist_threshold`                              | 0.55（语义建议 0.85）                   | `GeneralMemory(...)`；CLI `--gist-threshold` / `UI_GIST_THRESHOLD`（见 5.3）           | 相似判定门槛；语义场景调高挡住域外误召回         |
| `coherence_min`                               | 0.6                               | `GeneralMemory(...)`、`Hierarchy.compress()`                                        | 答案一致度门槛；低于此拒绝合并              |
| `min_confidence`                              | 0.6                               | `GeneralMemory(...)`、`Hierarchy(...)`；CLI `--min-confidence`                       | 输出授权门；None 关闭门控              |
| `semantic_weights`                            | None                              | `build_state(...)`；CLI `--semantic-weights` / `UI_SEMANTIC_WEIGHTS`（见 4.1.1 / 5.3） | 注入语义编码器权重 npz；不传用字符编码器       |
| `complexity_scale`                            | 3.0                               | `GeneralMemory(...)`                                                               | support → 复杂度的归一化尺度          |
| `gradual_decay`                               | True                              | `GeneralMemory(...)`（见 4.1.2）                                                      | 渐进退化开关（默认开）；False 退回旧式一步删    |
| `detail_threshold`                            | 0.5                               | `GeneralMemory(...)`（见 4.1.2）                                                      | 细节完整度低于此值才允许沉淀成要旨            |
| `tau` / `alpha`（浅记忆）                          | 3.0 / 1.0                         | `GeneralMemory(...)`、`EpisodicBuffer(...)`                                         | 浅记忆幂律衰减；也是逐字细节完整度的退化率        |
| `threshold` / `min_strength`                  | 0.20 / 0.0                        | `ForgettingProfile(...)`（`forgetting.py`）                                          | 保持率低于阈值即遗忘；查询时信任度下限          |
| `enable_forgetting`                           | True                              | `build_state(...)`；CLI `--no-forgetting`                                           | HTTP 服务是否注入遗忘画像              |
| `ALPHA` / `HALFLIFE_*`                        | 0.13 / 40min·4d·30d·25y           | `forgetting.py` 模块常量                                                               | 论文遗忘曲线的形状与四层半衰期              |
| `grid_size` / `extent`                        | 32 / 1.0                          | `PolicyConfig(...)`（`policy.py`）                                                   | 每维格数、状态覆盖范围                  |
| `n_skills` / `act_dim`                        | 1 / 2                             | `PolicyConfig(...)`                                                                | 技能数、动作维度                     |
| `max_turn_deg`                                | 8.0                               | `PolicyConfig(...)`、`RateLimiter(...)`                                             | 相邻输出的转角上限（度）                 |
| `hit_eps`                                     | 1e-3                              | `PolicyConfig(...)`                                                                | 判定完全命中（w≈1）的容差               |
| `min_episodes`（控制固化）                          | 5                                 | `CandidatePool(...)`（`consolidation/candidate.py`）                                 | 控制版固化门槛（与内容版独立）              |
| `memory_file`                                 | `checkpoints/general_memory.json` | `build_state(...)`；CLI `--memory-file` / `--no-persist`                            | 持久化文件；存在则启动恢复，写操作后落盘         |
| `persist_query_every`                         | 50                                | `build_state(...)`                                                                 | 命中统计的节流落盘间隔（查询次数）；停服时再 flush |
| `now`                                         | 0.0                               | 各方法参数 / HTTP 请求字段                                                                  | 记忆时间（天），用于衰减与遗忘              |

## 七、拓展与边界

**能力边界**：这是「背下来后脱口而出」，不是推理。`1+1=2` 背过能答，`137×842` 没背过答不出——现算需另接规则引擎。要旨联想是单向的（正向能接，反着问不出）。语义编码器（见 4.1.1）只学 KV 映射里出现过的等价问法，不是通用语义理解；零重叠的新说法需补进映射重训，通用语义要靠预训练大模型。

**可扩展方向**：

- **持久化**：已实现，`GeneralMemory.save/load` 与 HTTP 服务自动落盘（见 4.4）；多进程共享同一文件需自己加锁，当前面向单实例。
- **在线学习**：目前是离线练习→固化→只读。`Policy.end_episode()` 已预留失败回滚接口，继续学需加影子验证。
- **技能数扩展**：控制版技能数超过 3 个开始退化，更多技能建议分层（底层共享原语 + 上层选择器）。
- **高维输入**：格数随维度指数增长，超过 8 维应退化为纯网络或换表征。
- **端侧部署**：推理核心是哈希 + 查表 + 向量运算，可移植为原生库；语义编码器推理是 numpy 矩阵乘（无 torch），学习过程留在上位机。
- **语义等价扩展**：部署方按业务场景维护 KV 映射（同一答案的多种问法，长文本写法见 `examples/long_text_kv.json`），重训 `semantic.npz` 即可扩展等价类，无需改内核；映射越丰富，能接住的说法越多。

**依赖**：numpy（推理）、torch（仅离线训练，训练后导出 npz 即不再需要）、fastapi + uvicorn（HTTP 服务）、httpx（测试与客户端）。
