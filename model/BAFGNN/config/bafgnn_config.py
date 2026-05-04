from dataclasses import dataclass, field
from typing import List, Optional

from basicts.configs import BasicTSModelConfig


@dataclass
class BAFGNNConfig(BasicTSModelConfig):
    input_len: Optional[int] = field(default=None, metadata={"help": "Input sequence length."})
    output_len: Optional[int] = field(default=None, metadata={"help": "Output sequence length."})
    num_nodes: Optional[int] = field(default=None, metadata={"help": "Number of nodes/features."})

    in_channels: int = field(default=1, metadata={"help": "Input channels per node."})
    out_channels: int = field(default=1, metadata={"help": "Output channels per node."})

    embed_dim: int = field(default=64, metadata={"help": "Hidden embedding dimension."})
    num_heads: int = field(default=4, metadata={"help": "Attention heads for temporal encoder."})
    num_layers: int = field(default=2, metadata={"help": "Number of temporal encoder layers."})
    ff_dim: int = field(default=256, metadata={"help": "Feed-forward hidden dimension."})
    dropout: float = field(default=0.1, metadata={"help": "Dropout rate."})

    adj: Optional[List[List[float]]] = field(default=None, metadata={"help": "Static prior adjacency [N, N]."})
    bias_scale: float = field(default=1.0, metadata={"help": "Scale factor b for graph bias term."})
    attn_tau: float = field(default=1.0, metadata={"help": "Softmax temperature for graph-biased attention."})

    film_hidden_dim: int = field(default=64, metadata={"help": "Hidden size of FiLM MLP."})
