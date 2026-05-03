from dataclasses import dataclass, field
from typing import Optional

from basicts.configs import BasicTSModelConfig


@dataclass
class PDFormerConfig(BasicTSModelConfig):
    input_len: Optional[int] = field(default=None, metadata={"help": "Input sequence length."})
    output_len: Optional[int] = field(default=None, metadata={"help": "Output sequence length."})
    num_nodes: Optional[int] = field(default=None, metadata={"help": "Number of nodes/features."})

    input_dim: int = field(default=1, metadata={"help": "Input channels per node."})
    output_dim: int = field(default=1, metadata={"help": "Output channels per node."})

    embed_dim: int = field(default=64, metadata={"help": "Embedding dimension."})
    num_heads: int = field(default=4, metadata={"help": "Attention heads."})
    num_layers: int = field(default=3, metadata={"help": "Encoder layers."})
    ff_dim: int = field(default=256, metadata={"help": "Feed-forward hidden dimension."})
    dropout: float = field(default=0.1, metadata={"help": "Dropout rate."})
