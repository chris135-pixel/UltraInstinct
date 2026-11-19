# Ultra Instinct

A pure-software "muscle memory" system: well-practiced content is stored for direct lookup; unseen input is not answered on faith (the control kernel computes via the network, while the general-memory kernel falls back to shallow memory or refuses), and learning new things never overwrites old ones. The inference path has no deep-learning framework dependency and runs on a plain CPU.

[中文版本](README.md)

## Preface

With large models becoming ever more capable, I wanted a small, precise base model instead. The name comes from Ultra Instinct, the form of Son Goku in Dragon Ball Super.

It simulates the brain's muscle-memory cycle: repeated training moves an item from short-term to long-term memory; after prolonged disuse, detail fades into a gist; re-encounter regrows detail; after a very long time unused, it is deleted — a span aligned with the ~25-year scale of human memory.

The original intent was a narrow foundation model in a world of general giants, possibly as a fast subconscious front-end to a large model, with future use in robotics in mind.

On speed (with the default character encoder): a verbatim kernel hit takes ~18µs median (p99 ~36µs), a gist hit ~35µs (p50 still 35.5µs after deposition via `decay(now=30)` in the current kernel); through the full HTTP service, verbatim end-to-end is ~1.3ms median and the first streamed chunk ~25ms. With the local semantic encoder injected, short-text encoding is ~60µs p50; for ~200-character long text, feature extraction and one query take ~987µs and ~1008µs, with HTTP end-to-end p50 of 1.6~2.8ms — still millisecond-scale. For reference, conscious human reaction to a simple stimulus is on the order of tenths of a second: for practiced content this system reacts an order of magnitude faster than a person, almost without "thinking" — exactly what muscle memory means.

Its strengths are speed, frugality, stability and precision: practiced content is a table lookup, no deep-learning framework is needed, and a plain CPU suffices; default features are computed from character n-grams with no vocabulary or embedding model, while an optional locally written semantic net handles "different phrasing, same answer"; conflicting new content is separated rather than overwritten; "recalling a candidate" and "being allowed to state it" are two stages, so practiced content is authorized and unpracticed content is refused rather than fabricated. Memory can be saved and restored across processes.

Its limits come from the same mechanism: this is "recite from memory," not reasoning — a memorized `1+1=2` is answerable, an unmemorized `137×842` is not, and on-the-fly computation needs a separate rules engine; gist association is one-way (forward works, reverse doesn't); default similarity is literal n-gram based, so rephrasings may miss, and the optional semantic net only learns equivalences present in training data, not general semantics; learning is currently offline practice → consolidation → read-only, with no online learning; memory lives in one process, and multiple instances sharing one file need their own locking.

## 1. Concept

The core idea: **practice until you no longer have to think**. This has nothing to do with whether the stored content is a movement direction or a piece of text — the brain turns repetition into fast retrieval, and this system simulates that with one shared mechanism.

Three design goals follow:

1. **Practice makes permanent.** Content practiced repeatedly with consistent results is promoted to directly queryable memory and retrieved in O(1).
2. **Don't bluff when you don't know.** The control kernel computes via the general network; the general-memory kernel falls back to a shallow-memory guess or refuses — no answer is fabricated.
3. **Learn without overwriting.** Conflicting new content is separated from old memory rather than overwriting it; unused content is compressed into a gist, and details are recovered by re-observation when needed.

The system has two kernels sharing the same tiered-memory logic; only the key representation differs:

| Kernel                                                    | Input → Output                       | Use case                                             |
| --------------------------------------------------------- | ------------------------------------ | ---------------------------------------------------- |
| **General content** (`encoding.py` + `general_memory.py`) | any text/bytes → any text            | facts, scripts, poetry continuation, memorized Q&A   |
| **2D control** (`state.py` + `policy.py`)                 | `(Δx, Δy, skill)` → direction vector | robotic arm reaching, low-dimensional motion control |

There are three entry points; pick as needed: `GeneralMemory` (general memory Python API), `Policy` (control Python API), and `api.py` (OpenAI-compatible HTTP streaming API backed by the general memory kernel).

## 2. Design

### 2.1 Tiered memory: shallow → verbatim → gist

Memory has three tiers plus a write-back path, simulating "a few trials enter shallow memory, enough consistent trials enter deep memory, long-unused content compresses into a gist, and re-encounter restores detail":

```
practice → shallow memory (write-once, small, fast decay)
              ↓ repeated across episodes with consistent answers
          verbatim memory (full detail, exact lookup)
              ↓ long unused + consistent within block
          gist (core + local slope, much smaller)
              ↑
              └── re-encounter → regrow verbatim traces locally (reconsolidation)
```

Four key trade-offs:

- **A gist keeps the slope, not just the mean.** Compression means discarding second-order curvature while retaining first-order slope — not averaging details together.
- **Contradictory blocks are rejected** (pattern separation). Merging similar inputs with different answers averages into something correct for neither; they must be separated.
- **Detail recovery is re-observation, not decompression.** Verbatim and gist traces are stored in parallel; compression deletes the supporting points, so a gist cannot be expanded back. Reconsolidation can only regrow from new samples locally.
- **Forgetting is a feature.** Compressed content fades by complexity — detail-rich content goes first, core gist lasts — matching "generalization beats exact storage."

### 2.2 Fast and slow paths

This section describes the **2D control kernel's** read path. The general-content kernel does not mix with a network; it uses the three-tier fallback in 4.1 (verbatim → gist → shallow → refusal).

- **Write path (offline, may be slow):** practice → candidate pool → episode threshold → consistency check → table entry.
- **Read path (online, must be fast):** input → table lookup yields weight w → high w returns the table value, low w runs the network, in-between mixes them.

The slow path learns and generalizes to novel cases; the fast path handles fluent cases with zero decision-making. They are complementary: fast alone cannot generalize outside coverage, slow alone computes every time.

Mixing is not an either/or choice but a continuous interpolation by consolidation weight w: `mixed = w × table_value + (1−w) × net_value`. This is the single most important design finding in the project — measured at the consolidation boundary, hard switching gives an average angle of **96.2°** between the two instructions (max 103.7°, 100% over 30°), while continuous mixing gives **6.0°** (max 10.3°, 0% over 30°).

Two easy-to-hit pitfalls:

- **The weight must come from the interpolated occupancy mask, not "distance to the nearest cell center."** That distance is itself discontinuous at cell boundaries, so using it as a weight is no smoothing at all — the measured jump is still 90°.
- **Guard against degeneracy when the two paths nearly oppose.** As the angle approaches 180° the mixed vector's magnitude tends to zero, and normalizing it amplifies numerical noise into a random direction; `blend()` sets `degenerate_eps` (default 0.2) and skips normalization below it.

### 2.3 Accuracy first: two-stage decision

"Recalling a candidate" and "being allowed to state it" are separated, so a vague association is never treated as a confident answer:

- **Association** is decided by a similarity threshold — only whether a candidate exists.
- **Authorization** is decided by the confidence gate `min_confidence`. Fluent verbatim memory is always authorized; gist/shallow hits need `weight >= gate`.
- Non-strict mode returns the answer with a `confident` flag; strict mode (`strict=true`) returns empty for low-confidence hits, for execution-style use cases.

### 2.4 Key design decisions

The table below mainly comes from the 2D control kernel. The general-content kernel keeps cross-episode consolidation and numpy inference, but does not use bilinear interpolation or turn limiting.

| Decision                                                     | Reason                                                                                           |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| State uses relative `(Δ, skill)`, not absolute coordinates   | all targets share one table; relative quantities generalize                                      |
| State must carry a skill ID                                  | same position with different intent means opposite actions; without it, conflicts are inevitable |
| Bilinear interpolation for lookup, not nearest cell          | low resolution + interpolation beats high-resolution nearest neighbor, with less memory          |
| Mix weight comes from interpolated occupancy mask            | keeps boundaries continuous, avoids hard-switch jumps                                            |
| Consolidation counts across episodes                         | repeated trials in one episode don't count as multiple validations                               |
| Reliability via failure attribution, not up-front thresholds | better trade-off between coverage and accuracy                                                   |
| numpy for inference, not torch                               | removes framework dispatch overhead; several times faster per step                               |

## 3. Architecture

### 3.1 Layers

| Layer                  | Files                                                | Responsibility                                                                          |
| ---------------------- | ---------------------------------------------------- | --------------------------------------------------------------------------------------- |
| State/content encoding | `state.py`, `encoding.py`, `semantic.py`             | observation → `(Δ, skill)` or content → key + features (optional semantic encoding)     |
| Quantization & lookup  | `quantize.py`, `memory.py`                           | coordinates → cell index + interpolation weights; verbatim table query/undo/persistence |
| Tiered memory          | `hierarchy.py`, `general_memory.py`, `forgetting.py` | three tiers, compression, reconsolidation, complexity-based forgetting                  |
| Slow path              | `net.py`, `training/`                                | numpy MLP forward; offline torch training exporting npz                                 |
| Safety & routing       | `blend.py`, `safety.py`, `policy.py`                 | weighted mixing, turn limiting, failure attribution                                     |
| Service                | `api.py`                                             | OpenAI-compatible HTTP streaming API                                                    |

### 3.2 Directory

```
UltraInstinct/
├── ultra_instinct/
│   ├── state.py / encoding.py      state and character content encoding
│   ├── semantic.py                 injectable semantic encoder (local self-written net)
│   ├── quantize.py / memory.py     grid quantization and verbatim table
│   ├── hierarchy.py                tiered memory (control)
│   ├── general_memory.py           general content memory
│   ├── forgetting.py               layered forgetting (paper parameters)
│   ├── net.py / blend.py / safety.py  slow path, mixing, turn limiting
│   ├── policy.py                   control public API
│   ├── api.py                      HTTP streaming service
│   ├── training/                   offline training (torch: semantic_train.py semantics + trainer.py control)
│   └── consolidation/              consolidation pipeline
├── examples/                       five demos + KV/long-text training samples (see 4.1.1, 5.3)
├── checkpoints/                    semantic weights and memory files (weights and memory must match)
├── benchmarks/                     9 benchmark scripts + measured report (see 3.3)
├── tests/                          176 unit tests (semantics, API gate, gradual decay included)
├── configs/default.json            sample config (not loaded by code; defaults in §6)
└── requirements.txt                dependencies
```

### 3.3 Measured metrics

`benchmarks/bench_metrics.py` runs an end-to-end measurement; the report lands in `benchmarks/metrics_report.json`. Start the service first, then run the script:

```bash
python -m ultra_instinct.api --host 127.0.0.1 --port 8000
python benchmarks/bench_metrics.py
```

One typical run (training data is 18 simple items, yielding 6 verbatim entries; the kernel microbenchmark uses the default `ContentEncoder`, and the gist is deposited via `decay(now=30)` under the current default gradual decay):

| Metric                                   | Measured                                                        |
| ---------------------------------------- | --------------------------------------------------------------- |
| Hit rate on practiced items              | 100% (6/6)                                                      |
| Refusal rate on unseen items             | 100% (3/3)                                                      |
| Typo tolerance                           | "窗前明月" → "疑似地上霜" (shallow, weight 0.5)                          |
| Verbatim lookup p50 / p99                | 18.1µs / 36.4µs                                                 |
| Gist lookup p50 / p99                    | 35.5µs / 90.5µs                                                 |
| No-hit fallback p50                      | 137.3µs                                                         |
| HTTP verbatim end-to-end p50             | 1.32ms                                                          |
| SSE first chunk / total                  | 25.1ms / 25.7ms (7 chunks, reassembled correctly with `[DONE]`) |
| Footprint                                | verbatim 1542 + shallow 4644 = 6186 params                      |
| Gist half-life (complexity high/med/low) | 0.03 d / 1.07 d / 50.37 d                                       |

These numbers are a record of one real run and vary with machine and data — they indicate orders of magnitude, not a guarantee.

Supplementary measurements for semantics / long text (a separate dataset, not pooled with the table above):

| Metric                                                               | Measured                                                                    |
| -------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Synonym-group similarity (before → after training)                   | 0.154 → 0.996                                                               |
| Antonym-group similarity (after training)                            | −0.463                                                                      |
| Semantic short-text encoding p50                                     | 60µs (default character encoding 58µs)                                      |
| ~200-character long text: features / one query                       | ~987µs / 1008µs                                                             |
| Semantic service HTTP end-to-end p50                                 | 1.6~2.8ms                                                                   |
| Threshold 0.85: long-text hits / out-of-domain refusals (non-strict) | 8/8 synonyms return the same answer, 4/4 out-of-domain return empty         |
| Authorization state of short phrasings                               | `source=episodic`, `weight=0.5`, `confident=false` (blocked in strict mode) |

Eight more dedicated scripts live under `benchmarks/`, each answering one design question, with results in a same-named json alongside:

| Script               | Question it answers                                                        |
| -------------------- | -------------------------------------------------------------------------- |
| `bench_continual.py` | does learning new content overwrite old memory (learn without overwriting) |
| `bench_trust.py`     | hard-switch vs continuous-mixing jump angle at the consolidation boundary  |
| `bench_hierarchy.py` | coverage / footprint gain of tiered memory over a single layer             |
| `bench_safety.py`    | effect of quantization grid size and turn limiting on action error         |
| `bench_scaling.py`   | lookup-table vs KNN latency curve as capacity grows                        |
| `bench_torch.py`     | numpy forward vs torch single-step / batch latency                         |
| `bench_capacity.py`  | capacity and latency across network sizes                                  |
| `bench_latency.py`   | baseline latency breakdown of the numpy forward pass                       |

## 4. Core logic

### 4.1 General content: `GeneralMemory`

**Encoding** (`encoding.py`): the exact key is `blake2b(normalized_text)` truncated to 16 bytes, stable across processes; the feature vector is 256-dim L2-normalized from character 1/2/3-grams with the hashing trick, used for fuzzy matching; similarity is just the dot product of the normalized vectors (cosine). Character n-grams rather than a word vocabulary keep it friendly to Chinese ("窗前明月光" and "床前明月光" differ by one character yet stay close in vector space), avoid a fixed vocabulary, and keep the dimension from growing with the corpus.

Using `blake2b` rather than Python's built-in `hash()` is mandatory: `hash()` is salted randomly and changes every process start, so keys saved into a checkpoint would no longer match on the next launch. Normalization is NFKC + whitespace removal + lowercasing — it erases meaningless differences like full-width versus half-width forms but never touches substantive text.

**Read path** (`query()`, three tiers falling back):

1. Verbatim: exact key hit → return directly, `weight=1.0`, always authorized.
2. Gist: similarity to a gist center exceeds its coverage radius → return gist answer, `weight = similarity × coherence × retention`, authorized only if it passes the gate.
3. Shallow: vote among similar episodes → return the voted answer, `weight` capped at 0.5 inside `GeneralMemory.query()` (so unconsolidated content fails the default 0.6 gate).
4. No hit → return `None`; no answer is fabricated.

**A hit is not training**: `query()` only updates `hits` / `last_used` (keeping memory fresh, see 4.1.2); it does not write `_pending` and does not count as a cross-episode consolidation — a lookup does **not** advance the "short-term → long-term" consolidation progress, nor does it treat the matched phrasing as a new training sample. Progress comes only from `practice()` / `reconsolidate()`; queries through HTTP `/v1/chat/completions` never advance it. This avoids voting a tentative hit in as ground truth and locking a wrong answer in ever more firmly.

**Write path**: `practice()` writes to shallow memory (repeats within one episode count once) → `consolidate()` promotes to verbatim if the cross-episode threshold is met and the majority answer is stable → `compress()` clusters similar verbatim entries, merges only if answers agree, and truly deletes the verbatim entries (with default gradual decay, only entries with `detail ≤ detail_threshold` are processed; see 4.1.2) → `reconsolidate()` re-observes at gist hits and regrows verbatim traces.

`compress()` decides consistency by **majority vote within the cluster**: `coherence = votes for the most common answer / cluster size`, and a value below `coherence_min` rejects the merge (similar inputs with different answers would only average into something correct for neither — they must be separated). A gist's coverage radius is the **minimum similarity** of cluster members to the center, so the whole cluster is covered; the center is the mean of member features re-normalized, and complexity is normalized from support (cluster size) via `complexity_scale` and handed to the forgetting layer to decide the half-life.

#### 4.1.1 Semantic encoder (`semantic.py`, optional injectable)

The default `ContentEncoder` uses character n-grams and only sees literal overlap — "它是什么狗" and "那狗是啥色" are far apart literally and won't match. For deployments that need **different phrasings, same answer**, inject `SemanticEncoder`: a small locally self-written network that maps the n-gram vector to a "meaning vector", pulling synonymous phrasings together.

**Training**: write the deployment's KV map as `{answer: [synonymous phrasings, ...]}` (see `examples/kv_example.json`) and train with supervised contrastive loss — phrasings sharing an answer are positives, different answers are negatives:

```bash
python -m ultra_instinct.training.semantic_train \
    --pairs examples/kv_example.json --out checkpoints/semantic.npz
# Long-text equivalence classes (one answer with a long text + two short phrasings):
python -m ultra_instinct.training.semantic_train \
    --pairs examples/long_text_kv.json --out checkpoints/semantic_longtext.npz
```

`examples/long_texts.json` is the readable list of the same long texts; `examples/long_text_kv.json` is the actual training input.

Training uses torch and exports to `net.py`'s MLP npz format; the inference side is numpy-only, zero torch dependency. Measured: the synonym-group similarity rose from 0.154 to 0.996 while an antonym group was pushed to −0.463; **encoding latency rose only ~2µs over plain characters** (p50 58→60µs), because the bottleneck is n-gram hashing, not matrix multiplication.

**Usage**: untrained, `SemanticEncoder.features()` falls back to `ContentEncoder`'s character features (usable without training); note that `dim` defaults to 64 as a placeholder then, not the 256-dim character vector, and loaded weights override it. Once weights are loaded it injects directly with no kernel changes:

```python
from ultra_instinct import GeneralMemory, SemanticEncoder
enc = SemanticEncoder.from_weights("checkpoints/semantic.npz")
mem = GeneralMemory(enc)
mem.practice("它是什么颜色的狗", "它是一条黑色的德牧", 0)
mem.consolidate()
mem.query("那狗是啥色").response   # rephrased → semantic hit, same answer
```

The exact path is unchanged: literal inputs still hit verbatim via `blake2b` in microseconds; the semantic network only catches phrasings trained under the same answer.

Note that a semantic hit is not the same as an authorized hit: unseen phrasings may only reach shallow memory (`weight=0.5`, `confident=false`). Non-strict mode returns the same answer for reference, while strict mode blocks it at the authorization gate; to authorize such a phrasing for execution, practice it into verbatim memory too.

**Capability boundary**: the network only learns equivalences present in the training data — it is not general semantic understanding. "客户编号" and "用户ID" with zero overlap won't be brought together unless they appear under the same answer in the KV map and you retrain. That is the unavoidable trade for being lightweight, self-written, and microsecond-fast — general semantics needs a pretrained large model, which is not this project's bet.

**Persistence**: `save()` does not store network weights; only the encoder's `dim / ngrams / normalize` are saved alongside the full memory state in the same file. On restore you must still inject the same semantic encoder explicitly — `GeneralMemory.load(path, encoder=enc)` — otherwise it falls back to the default `ContentEncoder` and feature dimensions won't match.

#### 4.1.2 Gradual decay: the verbatim → gist transition (`gradual_decay`, on by default)

**Problem it solves**: early implementations made "verbatim → gist" rely on a single manual `compress()` call that erased all detail at once — a switch, not a process. The human brain first keeps newly mastered material with **full detail** for a long time, then, after a period of disuse, detail fades progressively into a gist; using it again in between refreshes it. This time-driven decay is therefore **enabled by default** — no explicit opt-in needed.

When on (`gradual_decay=True`, the default):

- Each verbatim entry carries a **detail completeness** `detail = (1 + age/τ)^(-α)`, `age = now − last_used`. It is 1.0 right after consolidation or use (full detail) and decays continuously with time.
- **Use refreshes detail**: a `query()` hit or `reconsolidate()` updates `last_used`, restoring detail to full — frequently used items stay crisp, unused ones blur.
- **Gradual deposition**: `decay(now)` (or `compress()`) merges only verbatim entries with `detail ≤ detail_threshold` into gists; entries still vivid remain in the verbatim tier. A batch of memories therefore fades at different times rather than all at once.
- **Re-consolidation restores detail**: a fresh observation regrows full detail (0.09 → 1.0) — it is not decompression from the gist.
- The time anchor uses the real `now` (`born`/`last_used` written at consolidation) and is persisted with the memory; the old "born stored as an ordinal" bug is fixed along the way.

**Off** (`gradual_decay=False`): falls back to the old semantics — `detail()` is always 1.0, `decay()` is a no-op, and `compress()` deletes in one immediate step. This switch exists only for backward compatibility with old callers and is not recommended to disable by default. Restoring from an older memory file (whose config lacks the field) also treats it as **enabled**.

```python
from ultra_instinct import GeneralMemory
m = GeneralMemory(gradual_decay=True, detail_threshold=0.5, tau=3.0, alpha=1.0)
# ... practice / consolidate ...
m.decay(now=30)     # advance time to "day 30"; only faded verbatim entries deposit
                    # entries hit by query() in between refresh last_used and are kept
# for old-style "erase in one step", opt out explicitly: GeneralMemory(gradual_decay=False)
```

### 4.2 Forgetting (`forgetting.py`)

Retention is `R(t) = (1 + t/τ)^(-α)`, with `α=0.13` fitted on a grid over Ebbinghaus's (1885) original seven data points (20min 58.2% / 1h 44.2% / 8.8h 35.8% / 1d 33.7% / 2d 27.8% / 6d 25.4% / 31d 21.1%), max error < 0.03 across all points.

**Why a power law and not an exponential**: exponential decay implies a characteristic time constant that does not match the data; Wixted & Ebbesen (1991) and Anderson & Schooler (1991) both support the power law. Note the data come from a single subject, meaningless syllables, and a relearning method — it is a **shape reference**, not a universal constant.

Memory strength is a weighted sum of four power-law components; complexity sets the weights — more detail gives more weight to short-lived components, so it fades faster. **The four layers are not four different curves** — α is held constant and the same Ebbinghaus curve is stretched horizontally (only τ changes, `R_layer(t) = R_eb(t×s)`):

| Component                 | Half-life | Source          |
| ------------------------- | --------- | --------------- |
| Verbatim / surface detail | 40 min    | Kintsch 1990    |
| Propositional / textbase  | 4 days    | Kintsch 1990    |
| Core gist                 | 30 days   | Sacripante 2023 |
| Semantic / permastore     | 25 years  | Bahrick 1984    |

Complexity-to-weight is a **linear map with no extra tunable parameter** (`c, (1−c)/3, (1−c)/3, (1−c)/3`), which makes "more complex forgets first" monotonically true for any `age>0` and directly unit-testable. The mechanism comes from the RAFT model (Fisher & Radvansky 2019): a gist retaining many verbatim details weights short-lived components heavily and drops first; a highly compressed core gist weights long-lived components and resists forgetting.

**Practice only changes the intercept, not the slope** (Slamecka & McElree 1983; Rivera-Lares 2022) — the number of practice trials acts on the initial strength, not on α / τ above, so "more practice" raises the starting point rather than slowing the forgetting rate.

`forget(now)` truly deletes gists whose retention falls below the threshold (off by default; enable by injecting `ForgettingProfile`). The time unit is **days**.

### 4.3 2D control: `Policy`

One `act()` decision: encode state → table lookup yields weight → high weight returns table value / low weight runs network / in-between mixes → turn limiting → return result (`action, weight, source, net_called, clamped`). `begin_episode()` / `end_episode()` delimit a task cycle; on failure, cells relied on during that cycle are undone.

**Why failure attribution is episode-level, not step-level**: two step-level criteria were tried and both failed — "distance didn't shrink means failure" killed 99.7% of decisions, because distance legitimately plateaus when going around an obstacle; "more than 5% worse than the best so far means regression" wrongly killed 52 of 118 correct memories. The root cause is that a 90° error keeps distance constant, so no distance-based process metric can detect "wrong direction." Using the definite signal "did this task succeed overall" to infer which steps' muscle memory is suspect is coarse-grained but does not misfire, and the attribution direction is correct: the muscle memory was wrong, hence the failure.

Relatedly, `RateLimiter`'s single-step turn limit is a backstop, not a replacement for mixing — after mixing, residual jumps reach 10.3° (reduced to `≤8.0°` via slerp); it is stateful, so `reset()` is required when switching tasks, otherwise the previous trajectory's direction drags the new task's first action.

**Training a control skill** (torch offline training; the exported model runs with zero torch dependency): `examples/reach_demo.py` is a complete four-step pipeline that produces `checkpoints/reach`.

1. **Train the slow path**: `TorchMLP` + `TrainConfig` + `train_skill()`, with a `ReplayBuffer` for joint replay so learning a new skill doesn't erase old ones; `export_mlp()` converts it to `net.py`'s numpy MLP.
2. **Collect successful experience**: run episodes and feed successful `(delta, action)` pairs to `CandidatePool.record()`; entries are aggregated per grid cell and counted across episodes.
3. **Compile into a table**: `pool.promoted()` selects cells that succeeded across `min_episodes` (default 5) episodes with action spread (`max_spread`, default 0.25) within bounds; `compile_memory()` compiles them into a `LUTMemory`.
4. **Assemble and save**: `Policy(net=net, memory=memory, config=...)`, then `save()` writes the trio `config.json` / `net.npz` / `memory.npz`.

```python
import numpy as np
from ultra_instinct.training import TorchMLP, TrainConfig, ReplayBuffer, train_skill, export_mlp
from ultra_instinct.consolidation import CandidatePool, compile_memory
from ultra_instinct.quantize import Grid
from ultra_instinct import Policy, PolicyConfig

rng = np.random.default_rng(0)
net_t = TorchMLP(input_dim=3, hidden=64, output_dim=2)  # 2D relative target + 1 skill code
cfg = TrainConfig(steps=1500, lr=2e-3, batch=256)
train_skill(net_t, task_fn, cfg, rng, skill_id=0, n_skills=1, replay=ReplayBuffer())
net = export_mlp(net_t)

pool = CandidatePool(Grid(32, 1.0), n_skills=1, min_episodes=5)
# ... run episodes: pool.record(delta, skill_id=0, action, episode_id)
memory = compile_memory(pool.promoted(), Grid(32, 1.0), n_skills=1, act_dim=2)
Policy(net=net, memory=memory, config=PolicyConfig()).save("checkpoints/reach")
```

`task_fn` comes from the scenario (`reach_demo.py` uses an ideal vector field for 2D reaching). The skill ID must be fed into the network input and vary within a batch, otherwise the network ignores that dimension (see the `training/trainer.py` module docstring; the inference side concatenates it automatically via `encode_state`).

### 4.4 Persistence (`save` / `load`)

`GeneralMemory.save(path)` writes the **complete state** to a single JSON file (byte keys and vectors are base64-encoded): all three tiers, the `_pending` consolidation counters, the caches still needed for pending items, every constructor parameter (including `gradual_decay` / `detail_threshold`), verbatim `born/last_used` and pending time anchors, **and the encoder config (dim / ngrams / normalize)** — keys and features come from the encoder, so omitting its config causes shape-mismatch crashes or normalize inconsistencies that silently return no answer. Writes are atomic (temp file + replace), so a crash never leaves a half-written file. `GeneralMemory.load(path)` resumes exactly where it left off — including continuing to consolidate unfinished pending items. Legacy files missing the encoder or gradual-decay fields fall back to the default encoder and default-on decay respectively (see 4.1.2).

```python
mem.save("checkpoints/general_memory.json")
mem2 = GeneralMemory.load("checkpoints/general_memory.json")
# A non-default encoder is restored from the file, no need to pass it again:
# m = GeneralMemory(ContentEncoder(dim=64, normalize=False)); m.save(path)
# GeneralMemory.load(path).encoder.dim == 64 and .normalize == False
```

Caches only serve re-consolidation of uncommitted items: after every `consolidate` / `compress` / `reconsolidate`, unreferenced cache entries are pruned so they never grow without bound into the file.

The HTTP service enables persistence by default: **both startup paths** (`python -m` and `uvicorn ultra_instinct.api:app`, the latter via lifespan) restore and save. Saves happen after each practice/compression; query hit statistics are saved throttled every `persist_query_every` (default 50) queries, plus one final flush on shutdown — keeping the microsecond hot path disk-free without losing counts on a clean stop. Memory has been verified to survive a cross-process restart intact.

HTTP does **not** expose `decay(now)` yet: `/v1/memories` calls `compress` with `now=0`, so freshly consolidated verbatim entries are not deposited under default gradual decay. To advance time-based decay, use the Python API `m.decay(now)` (see 4.1.2).

## 5. Usage

### 5.1 Install & run

```bash
pip install -r requirements.txt

# General content demo (1+1, poetry, typo tolerance, compression, reconsolidation, forgetting)
python examples/general_memory_demo.py
# Semantic encoder demo (KV-map training → rephrasing hits the same answer)
python examples/semantic_demo.py
# Long-text semantic training (training input: examples/long_text_kv.json)
python -m ultra_instinct.training.semantic_train \
  --pairs examples/long_text_kv.json --out checkpoints/semantic_longtext.npz
# 2D control demo (torch-train the slow path → compile memory table → save to checkpoints/reach)
python examples/reach_demo.py
# HTTP client demo (start the service first, see 5.3)
python examples/api_client_demo.py
# Authorization gate check: default vs strict (start the service first)
python examples/confidence_gate_client.py
# End-to-end measurement (start the service first; writes benchmarks/metrics_report.json)
python benchmarks/bench_metrics.py
# Unit tests
python -m unittest discover -s tests -t .
```

### 5.2 Python API

```python
from ultra_instinct import ContentEncoder, GeneralMemory
from ultra_instinct.forgetting import ForgettingProfile

mem = GeneralMemory(ContentEncoder(), min_episodes=3,
                    forgetting=ForgettingProfile())  # omit to disable forgetting
mem.practice("1+1", "2", episode_id=0)   # practice; use different episode_ids
mem.consolidate()                        # consolidate: promote cross-episode matches to verbatim
mem.compress(now=10)                     # optional: compress into gist (now in days)
answer = mem.query("1+1").response       # '2'
mem.reconsolidate("1+1", "2", episode_id=3, now=11)  # on a gist hit, regrow verbatim from a new observation
mem.forget(now=100)                      # optional: delete forgotten gists
```

Control kernel:

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
# optional: --min-confidence 0.6 (authorization gate), --no-forgetting (disable forgetting)
#           --memory-file <path> (default checkpoints/general_memory.json; auto-restore if present)
#           --no-persist (disable persistence, back to in-memory only)
#           --semantic-weights <npz> (inject semantic encoder, see 4.1.1)
#           --gist-threshold 0.55 (similarity gate; use 0.85 with semantics to block out-of-domain hits)
# With a direct uvicorn launch, set UI_SEMANTIC_WEIGHTS / UI_GIST_THRESHOLD instead
```

Start with the semantic encoder (rephrasing hits the same answer; a higher threshold avoids false recalls):

```bash
python -m ultra_instinct.api --host 127.0.0.1 --port 8000 \
  --semantic-weights checkpoints/semantic_longtext.npz --gist-threshold 0.85 \
  --memory-file checkpoints/semantic_test_memory.json
# Note: semantic weights and the memory file share a feature space and must match;
# use a fresh memory file when changing weights.
```

Test long text directly (after starting the service with the command above):

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"大脑怎么把短期记忆变成长期记忆"}]}'
# → "记忆会从短期巩固到长期，细节被剥离成要旨，需要时再重建"
# usage: source=episodic, weight=0.5, confident=false (short phrasing voted from shallow memory)

curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"股票今天涨不涨"}]}'
# → empty content, usage.source=none (out-of-domain input is not a hit; no answer fabricated)
```

For execution-style use with `"strict":true`, the short phrasings above are blocked by the authorization gate; practice those phrasings into verbatim memory first.

| Endpoint               | Method | Purpose                                                                                                 |
| ---------------------- | ------ | ------------------------------------------------------------------------------------------------------- |
| `/health`              | GET    | health check                                                                                            |
| `/v1/models`           | GET    | model list (id = `ultra-instinct`)                                                                      |
| `/v1/memories`         | POST   | practice; `consolidate` promotes, `compress` request flag (default false, called with `now=0`; see 4.4) |
| `/v1/memories/stats`   | GET    | memory footprint and counters                                                                           |
| `/v1/chat/completions` | POST   | chat; `stream=true` for SSE streaming                                                                   |

Practice (each item must reach `min_episodes` with distinct `episode_id`s):

```bash
curl -s http://127.0.0.1:8000/v1/memories -H 'Content-Type: application/json' \
  -d '{"items":[
        {"content":"1+1","response":"2","episode_id":0,"now":0},
        {"content":"1+1","response":"2","episode_id":1,"now":1},
        {"content":"1+1","response":"2","episode_id":2,"now":2}
      ],"consolidate":true}'
```

Query (non-streaming / streaming):

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"1+1"}]}'

curl -N http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"窗前明月光"}],"stream":true}'
```

Non-streaming returns a standard `chat.completion`; `usage` extends with `source` / `weight` / `confident`. Add `"strict":true` to return empty for low-confidence hits. The official OpenAI SDK works by pointing `base_url` at `http://127.0.0.1:8000/v1` (see `examples/api_client_demo.py`).

## 6. Core parameters and where to set them

All parameters have defaults; override them at **construction time** as needed. All time units are **days**.

| Parameter                                     | Default                           | Where to set                                                                           | Effect                                                                            |
| --------------------------------------------- | --------------------------------- | -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `dim` / `ngrams` / `normalize`                | 256 / (1,2,3) / True              | `ContentEncoder(...)` (`encoding.py`)                                                  | feature dimension, n-gram lengths, text normalization                             |
| `steps` / `hidden` / `out_dim` / `tau` / `lr` | 800 / 64 / 64 / 0.1 / 3e-3        | `semantic_train.py` CLI or `train_semantic(...)` (see 4.1.1)                           | semantic net training steps, hidden/out dims, contrast temperature, learning rate |
| `batch` / `seed`                              | 0 / 0                             | `semantic_train.py` CLI or `train_semantic(...)`                                       | batch size (0 = full batch), random seed for reproducibility                      |
| `min_episodes`                                | 3                                 | `GeneralMemory(...)`, `Hierarchy(...)`; HTTP `min_episodes`                            | episodes required before promotion to verbatim                                    |
| `stability`                                   | 0.6                               | `GeneralMemory(...)`                                                                   | majority-answer ratio required for consolidation                                  |
| `capacity`                                    | 64                                | `GeneralMemory(...)`, `EpisodicBuffer(...)`                                            | shallow memory capacity                                                           |
| `gist_threshold`                              | 0.55 (use 0.85 w/ semantics)      | `GeneralMemory(...)`; CLI `--gist-threshold` / `UI_GIST_THRESHOLD` (see 5.3)           | similarity gate; raise it with semantics to block out-of-domain hits              |
| `coherence_min`                               | 0.6                               | `GeneralMemory(...)`, `Hierarchy.compress()`                                           | answer-consistency gate; reject merge below it                                    |
| `min_confidence`                              | 0.6                               | `GeneralMemory(...)`, `Hierarchy(...)`; CLI `--min-confidence`                         | authorization gate; None disables gating                                          |
| `semantic_weights`                            | None                              | `build_state(...)`; CLI `--semantic-weights` / `UI_SEMANTIC_WEIGHTS` (see 4.1.1 / 5.3) | inject semantic encoder npz; omitted means character encoder                      |
| `complexity_scale`                            | 3.0                               | `GeneralMemory(...)`                                                                   | normalization scale from support to complexity                                    |
| `gradual_decay`                               | True                              | `GeneralMemory(...)` (see 4.1.2)                                                       | gradual-decay switch (on by default); False restores old one-step erase           |
| `detail_threshold`                            | 0.5                               | `GeneralMemory(...)` (see 4.1.2)                                                       | deposit to gist only when detail completeness falls below it                      |
| `tau` / `alpha` (shallow)                     | 3.0 / 1.0                         | `GeneralMemory(...)`, `EpisodicBuffer(...)`                                            | shallow-memory power-law decay; also the verbatim detail decay rate               |
| `threshold` / `min_strength`                  | 0.20 / 0.0                        | `ForgettingProfile(...)` (`forgetting.py`)                                             | retention below threshold is forgotten; minimum trust factor at query time        |
| `enable_forgetting`                           | True                              | `build_state(...)`; CLI `--no-forgetting`                                              | whether the HTTP service injects a forgetting profile                             |
| `ALPHA` / `HALFLIFE_*`                        | 0.13 / 40min·4d·30d·25y           | `forgetting.py` module constants                                                       | paper curve shape and four component half-lives                                   |
| `grid_size` / `extent`                        | 32 / 1.0                          | `PolicyConfig(...)` (`policy.py`)                                                      | cells per dimension, state coverage                                               |
| `n_skills` / `act_dim`                        | 1 / 2                             | `PolicyConfig(...)`                                                                    | number of skills, action dimension                                                |
| `max_turn_deg`                                | 8.0                               | `PolicyConfig(...)`, `RateLimiter(...)`                                                | max turn angle between outputs (degrees)                                          |
| `hit_eps`                                     | 1e-3                              | `PolicyConfig(...)`                                                                    | tolerance for treating a hit as exact (w≈1)                                       |
| `unit_action`                                 | True                              | `PolicyConfig(...)`                                                                    | normalize action to a unit vector (for direction-type actions)                    |
| `hidden` / `steps` / `lr` / `batch` (control train) | 64 / 1200 / 2e-3 / 256       | `TrainConfig(...)` (`training/model.py`, see 4.3)                                       | control slow-path training hyperparameters; `reach_demo.py` overrides steps=1500  |
| `replay_ratio` / `out_tanh` (control train)   | 0.5 / True                        | `TrainConfig(...)`                                                                     | old-skill replay ratio (anti-forgetting); tanh on output layer                    |
| `capacity` (replay buffer)                    | 200000                            | `ReplayBuffer(...)` (`training/trainer.py`)                                             | joint-replay sample cap                                                           |
| `min_episodes` (control)                      | 5                                 | `CandidatePool(...)` (`consolidation/candidate.py`)                                    | control-side consolidation threshold (separate from content)                      |
| `max_spread` (control)                        | 0.25                              | `CandidatePool(...)`                                                                   | action-spread limit; above it the cell is ambiguous and not consolidated         |
| `memory_file`                                 | `checkpoints/general_memory.json` | `build_state(...)`; CLI `--memory-file` / `--no-persist`                               | persistence file; auto-restore on startup, save after writes                      |
| `persist_query_every`                         | 50                                | `build_state(...)`                                                                     | throttled hit-stat save interval (query count); flushed on shutdown               |
| `now`                                         | 0.0                               | method args / HTTP fields                                                              | memory time (days), for decay and forgetting                                      |

## 7. Extensions and boundaries

**Capability boundary**: this is "recite it fluently from memory," not reasoning. A memorized `1+1=2` is answerable; an unmemorized `137×842` is not — on-the-fly computation needs a separate rules engine. Gist association is one-way (forward works, reverse doesn't). The semantic encoder (see 4.1.1) only learns equivalences present in the KV map, not general semantics; zero-overlap new phrasings need to be added to the map and retrained, and general semantics requires a pretrained large model.

**Extension directions**:

- **Persistence**: implemented — `GeneralMemory.save/load` and automatic disk saves in the HTTP service (see 4.4). Multiple processes sharing one file need their own locking; currently designed for a single instance.
- **Online learning**: currently offline practice → consolidation → read-only. `Policy.end_episode()` already reserves a failure-rollback hook; continued learning needs shadow validation.
- **More skills**: the control kernel degrades past ~3 skills; for more, use a hierarchy (shared low-level primitives + a high-level selector).
- **High-dimensional input**: cell count grows exponentially with dimensions; beyond 8 dimensions fall back to a pure network or change representation.
- **On-device deployment**: the inference core is hashing + lookup + vector math and can be ported to a native library; the semantic encoder infers with numpy matmul (no torch); keep learning on a host machine.
- **Semantic-equivalence extension**: deployments maintain a KV map (multiple phrasings per answer; see `examples/long_text_kv.json` for the long-text form) for their domain and retrain `semantic.npz` to grow the equivalence classes without touching the kernel; the richer the map, the more phrasings it catches.

**Dependencies**: numpy (inference), torch (offline training only; after export to npz it is no longer needed), fastapi + uvicorn (HTTP service), httpx (tests and client).
