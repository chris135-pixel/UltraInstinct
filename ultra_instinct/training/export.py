"""导出：torch 权重 → numpy npz。

torch 的 Linear 权重是 (out, in)，numpy 前向要 (in, out)，导出时转置。
之后推理路径就再也用不到 torch 了。
"""

from __future__ import annotations

import numpy as np

from ultra_instinct.net import MLP
from ultra_instinct.training.model import TorchMLP


def export_mlp(net: TorchMLP, path: str | None = None) -> MLP:
    """把训练好的 torch 网络转成 numpy 前向。

    Args:
        net: 已训练的 TorchMLP。
        path: 可选的保存路径（.npz）。

    Returns:
        可直接用于推理的 MLP。
    """
    sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    mlp = MLP(
        w1=np.ascontiguousarray(sd["body.0.weight"].T, dtype=np.float32),
        b1=np.ascontiguousarray(sd["body.0.bias"], dtype=np.float32),
        w2=np.ascontiguousarray(sd["body.2.weight"].T, dtype=np.float32),
        b2=np.ascontiguousarray(sd["body.2.bias"], dtype=np.float32),
        w3=np.ascontiguousarray(sd["body.4.weight"].T, dtype=np.float32),
        b3=np.ascontiguousarray(sd["body.4.bias"], dtype=np.float32),
        out_tanh=bool(net.out_tanh),
    )
    if path is not None:
        mlp.save(path)
    return mlp
