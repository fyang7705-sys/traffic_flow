from dataclasses import dataclass, field
from typing import List, Optional

from basicts.configs import BasicTSModelConfig


@dataclass
class HimNetConfig(BasicTSModelConfig):
    input_len: Optional[int] = field(default=None)
    output_len: Optional[int] = field(default=None)
    num_nodes: Optional[int] = field(default=None)
    input_dim: int = field(default=3)
    output_dim: int = field(default=1)
    hidden_dim: int = field(default=64)
    num_layers: int = field(default=1)
    cheb_k: int = field(default=2)
    ycov_dim: int = field(default=2)
    tod_embedding_dim: int = field(default=8)
    dow_embedding_dim: int = field(default=8)
    node_embedding_dim: int = field(default=16)
    st_embedding_dim: int = field(default=16)
    tf_decay_steps: int = field(default=4000)
    use_teacher_forcing: bool = field(default=False)
    time_of_day_size: int = field(default=288)
    use_time_embedding: bool = field(default=True)
    adj: Optional[List[List[float]]] = field(default=None)
