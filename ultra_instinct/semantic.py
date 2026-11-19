"""语义编码器：本地自写网络，把字符 n-gram 输入映射成「意思向量」。

能力边界（必读，别拿它当通用语义理解）：
    网络只能利用训练数据里同义问法之间的字词重叠，学到「哪些 n-gram 组合
    在这个任务里等价」。零重叠的两种说法（「客户编号」vs「用户ID」）不会
    被凭空拉到一起——输入里没有任何共同信号可供网络利用。
    它有真实但有限的泛化：训练里同义组足够丰富时，没见过、但有字词桥梁的
    问法也能落到同一簇。它不理解任意自然语言，那需要预训练大模型。

设计（沿用 net.py 的实测结论：训练用 torch，推理零 torch 依赖）：

    ┌ 精确路径：blake2b(原文) → 逐字 O(1) 命中，微秒，输入一字不差时走这里
    └ 语义路径：字符 n-gram 向量 → 自写 MLP → L2 归一 → cosine 命中变体

    - base 就是 ContentEncoder，key_of 与 clean 完全委托给它，精确键行为不变。
    - features 默认(=未加载权重)退化为 base 的字符特征，保证「不训练也能用」；
      set_weights/load_weights 注入 MLP 后才具备语义泛化。
    - 可注入：实现 encode/similarity/dim/ngrams/normalize，直接喂 GeneralMemory。

持久化注意：
    GeneralMemory.save() 只存 encoder 的 dim/ngrams/normalize 三项配置，
    不存网络权重；完整记忆状态另存同一文件。
    语义权重请用 save_weights() 单独存 npz；恢复时
    GeneralMemory.load(path, encoder=SemanticEncoder.from_weights(npz))。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ultra_instinct.encoding import ContentEncoder, Encoded
from ultra_instinct.net import MLP


class SemanticEncoder:
    """可注入 GeneralMemory 的语义编码器（精确键 + 网络语义向量）。

    Args:
        base: 基础字符编码器，提供稳定的 key_of/clean 与 n-gram 输入向量。
            不传则建一个 dim=256 的默认 ContentEncoder。
        dim: 语义向量输出维度（MLP 输出层宽度）。只在未加载权重时用于占位；
            加载权重后以权重为准。
        mlp: 已训练好的推理网络（ultra_instinct.net.MLP）。None 表示未训练，
            features 退化为纯字符特征。
    """

    kind = "semantic"

    def __init__(self, base: ContentEncoder | None = None,
                 dim: int = 64, mlp: MLP | None = None) -> None:
        self.base = base or ContentEncoder(dim=256)
        self._mlp = mlp
        self._dim = int(mlp.output_dim) if mlp is not None else int(dim)
        if mlp is not None and mlp.input_dim != self.base.dim:
            raise ValueError(
                f"网络输入维度 {mlp.input_dim} 与 base 特征维度 "
                f"{self.base.dim} 不一致")

    # ------------------------------------------------ GeneralMemory 需要的接口

    @property
    def dim(self) -> int:
        """feat 维度（语义向量宽度），GeneralMemory 用它算占用与存盘。"""
        return self._dim

    @property
    def ngrams(self) -> tuple[int, ...]:
        return self.base.ngrams

    @property
    def normalize(self) -> bool:
        return self.base.normalize

    @property
    def trained(self) -> bool:
        return self._mlp is not None

    def clean(self, content: str | bytes) -> str:
        return self.base.clean(content)

    def key_of(self, content: str | bytes) -> bytes:
        return self.base.key_of(content)

    def features(self, content: str | bytes) -> np.ndarray:
        """语义向量。未加载权重时退化为 base 的字符特征。"""
        x = self.base.features(content)          # 已 L2 归一
        if self._mlp is None:
            return x
        y = self._mlp.forward(x)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        nrm = float(np.linalg.norm(y))
        if nrm > 1e-12:
            y = y / nrm
        return y

    def encode(self, content: str | bytes) -> Encoded:
        text = self.base.clean(content)
        return Encoded(
            key=self.base.key_of(text),
            feat=self.features(text),
            text=text,
        )

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """向量已 L2 归一，点积即余弦。"""
        return float(np.clip(np.dot(a, b), -1.0, 1.0))

    # ------------------------------------------------ 权重管理

    def set_weights(self, mlp: MLP) -> None:
        """注入推理网络。校验输入维度，防止与 base 错配。"""
        if mlp.input_dim != self.base.dim:
            raise ValueError(
                f"网络输入维度 {mlp.input_dim} 与 base 特征维度 "
                f"{self.base.dim} 不一致")
        self._mlp = mlp
        self._dim = mlp.output_dim

    def save_weights(self, path: str) -> None:
        """把网络权重存成 npz。未训练时抛错，避免存出空文件。"""
        if self._mlp is None:
            raise ValueError("编码器尚未训练，没有权重可存")
        self._mlp.save(str(path))

    def load_weights(self, path: str) -> None:
        self.set_weights(MLP.load(str(path)))

    @classmethod
    def from_weights(cls, path: str, base: ContentEncoder | None = None) \
            -> "SemanticEncoder":
        enc = cls(base=base)
        enc.load_weights(path)
        return enc

    def __repr__(self) -> str:
        state = f"trained out_dim={self._dim}" if self.trained else "untrained"
        return f"SemanticEncoder({state}, base_dim={self.base.dim})"
