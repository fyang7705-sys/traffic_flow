from dataclasses import dataclass, field
from typing import Any, List, Optional

from basicts.configs import BasicTSModelConfig


@dataclass
class HimNetConfig(BasicTSModelConfig):
    # ----- data / shapes -----
    input_len: Optional[int] = field(default=None)
    output_len: Optional[int] = field(default=None)
    num_nodes: Optional[int] = field(default=None)

    # ----- io dims -----
    input_dim: int = field(default=1)
    output_dim: int = field(default=1)

    # ----- model (graph + recurrent) -----
    hidden_dim: int = field(default=64)
    num_layers: int = field(default=1)
    cheb_k: int = field(default=2)

    # ----- embeddings -----
    # time embedding dim = tod_embedding_dim + dow_embedding_dim
    tod_embedding_dim: int = field(default=8)
    dow_embedding_dim: int = field(default=8)
    node_embedding_dim: int = field(default=16)

    # node embedding 共享底座维度（用于 decouple/projection）
    node_base_dim: Optional[int] = field(default=None)
    decouple_node_embedding: bool = field(default=True)

    # ----- time-related -----
    time_of_day_size: int = field(default=288)
    use_time_embedding: bool = field(default=True)

    # SimpleTimeEmbedding (MLP-Mixer 风格)
    # default: if None, will fallback to input_len
    in_steps: Optional[int] = field(default=None)
    # 注意：实现里默认 time_d_model=32，为了对齐这里也用 32
    time_d_model: int = field(default=32)
    time_dropout: float = field(default=0.1)

    # ----- static graph prior -----
    # 单个静态邻接（训练脚本里通常传 np.ndarray.tolist()）
    adj: Optional[List[List[float]]] = field(default=None)
    # 额外静态先验（可选，元素可以是 list 或 numpy 数组；会在 arch 中转为 tensor）
    extra_static_supports: Optional[List[Any]] = field(default=None)
    use_graph_fusion: bool = field(default=True)
