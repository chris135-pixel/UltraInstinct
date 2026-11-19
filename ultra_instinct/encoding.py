"""通用内容编码：任意文本 / 字节 → 稳定的键 + 特征向量。

这是把原来「二维坐标」内核改造成「任意内容」内核的第一层。

设计约束（实测驱动，见 README 与 benchmarks/general_bench.py）：

1. 精确键必须稳定、无碰撞风险：
   用 blake2b 而不是 Python 内置 hash —— 后者带随机盐，跨进程会变，
   存进检查点的键下次启动就对不上了。

2. 特征向量用字符 n-gram + hashing trick，而不是词表：
   - 词表要提前固定，「1+1=2」和诗句接龙共用一个词表不现实；
   - 字符级对中文友好（「窗前明月光」和「床前明月光」只差一个字，
     向量仍然接近，能触发模糊命中）；
   - hashing trick 让维度固定，不随见过多少内容增长。

3. 归一化必须保留信息又消除无意义差异：
   全角半角、大小写、多余空白要抹掉，但「明月光」三个字不能动。

零外部依赖，只用了标准库 hashlib / unicodedata 和 numpy。
"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Encoded:
    """一段内容的两种表示。

    Attributes:
        key: 精确键（归一化内容的 blake2b 摘要），用于逐字记忆的 O(1) 命中。
        feat: 字符 n-gram 特征向量（L2 归一化），用于要旨层的模糊相似度。
        text: 归一化后的原文，用于输出与审计。
    """

    key: bytes
    feat: np.ndarray
    text: str


class ContentEncoder:
    """把任意文本/字节编码成精确键 + 特征向量。

    Args:
        dim: 特征向量维度。默认 256，对几千条规模的记忆足够，
             hash 碰撞对相似度的影响可忽略。
        ngrams: 使用的字符 n-gram 长度。中文 1/2/3 覆盖单字、词、短语。
        normalize: 是否做 NFKC + 去空白 + 小写。
    """

    def __init__(self, dim: int = 256, ngrams: tuple[int, ...] = (1, 2, 3),
                 normalize: bool = True) -> None:
        self.dim = int(dim)
        self.ngrams = tuple(ngrams)
        self.normalize = bool(normalize)

    def clean(self, content: str | bytes) -> str:
        """归一化：抹掉无意义差异，保留实质内容。"""
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="replace")
        else:
            text = str(content)
        if self.normalize:
            text = unicodedata.normalize("NFKC", text)
            text = "".join(text.split())   # 去掉所有空白（含全角空格）
            text = text.lower()
        return text

    def key_of(self, content: str | bytes) -> bytes:
        """精确键。 blake2b 跨进程稳定，可安全持久化。"""
        return hashlib.blake2b(
            self.clean(content).encode("utf-8"), digest_size=16).digest()

    def _hash_bucket(self, gram: str) -> int:
        """把 n-gram 稳定地映射到 [0, dim)。"""
        h = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, "little") % self.dim

    def features(self, content: str | bytes) -> np.ndarray:
        """字符 n-gram 的 hashing-trick 向量，L2 归一化。"""
        text = self.clean(content)
        vec = np.zeros(self.dim, dtype=np.float32)
        if not text:
            return vec
        for n in self.ngrams:
            if len(text) < n:
                continue
            for i in range(len(text) - n + 1):
                vec[self._hash_bucket(text[i:i + n])] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            vec /= norm
        return vec

    def encode(self, content: str | bytes) -> Encoded:
        """一次算出键、特征、归一化文本。"""
        text = self.clean(content)
        return Encoded(
            key=hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest(),
            feat=self.features(text),
            text=text,
        )

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """余弦相似度。特征已归一化，点积即可。"""
        return float(np.clip(np.dot(a, b), -1.0, 1.0))
