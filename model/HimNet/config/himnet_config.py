from dataclasses import dataclass, field
from typing import List, Optional

from basicts.configs import BasicTSModelConfig


@dataclass
class HimNetConfig(BasicTSModelConfig):
    # ----- data / shapes -----
    input_len: Optional[int] = field(default=None)
    output_len: Optional[int] = field(default=None)
    num_nodes: Optional[int] = field(default=None)

    # x feature dims
    input_dim: int = field(default=1)
    output_dim: int = field(default=1)

    # ----- model (graph+temporal) -----
    hidden_dim: int = field(default=64)
    num_layers: int = field(default=1)
    cheb_k: int = field(default=2)

    # decoder future cov dims
    ycov_dim: int = field(default=0)

    # embeddings
    tod_embedding_dim: int = field(default=8)
    dow_embedding_dim: int = field(default=8)
    node_embedding_dim: int = field(default=16)
    st_embedding_dim: int = field(default=16)

    # training tricks
    tf_decay_steps: int = field(default=4000)
    use_teacher_forcing: bool = field(default=False)

    # time-related
    time_of_day_size: int = field(default=288)
    use_time_embedding: bool = field(default=True)

    # ----- transformer (HimGCRU内置 Transformer) -----
    transformer_nhead: int = field(default=4)
    transformer_layers: int = field(default=1)
    transformer_ff_dim: Optional[int] = field(default=None)
    transformer_dropout: float = field(default=0.1)

    # ----- static graph prior -----
    adj: Optional[List[List[float]]] = field(default=None)
    use_graph_fusion: bool = field(default=True)
    
    # ----- iTransformer global time embedding (ITransformerGlobalTimeEmbedding) -----
    in_steps: Optional[int] = field(default=None)
    time_d_model: int = field(default=64)
    time_nhead: int = field(default=4)
    time_layers: int = field(default=1)
    time_ff_dim: Optional[int] = field(default=None)
    time_dropout: float = field(default=0.1)
