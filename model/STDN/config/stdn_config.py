from dataclasses import dataclass, field
from typing import List, Optional

from basicts.configs import BasicTSModelConfig


@dataclass
class STDNConfig(BasicTSModelConfig):
    input_len: Optional[int] = field(default=None, metadata={"help": "Input sequence length."})
    output_len: Optional[int] = field(default=None, metadata={"help": "Output sequence length."})
    num_nodes: Optional[int] = field(default=None, metadata={"help": "Number of nodes/features."})

    time_of_day_size: int = field(default=288, metadata={"help": "Time slots per day."})
    bn_decay: float = field(default=0.1, metadata={"help": "BatchNorm momentum."})

    L: int = field(default=2, metadata={"help": "Decoder layers."})
    K: int = field(default=16, metadata={"help": "Attention heads."})
    d: int = field(default=8, metadata={"help": "Hidden dim per head."})
    reference: int = field(default=3, metadata={"help": "Reference set size."})
    order: int = field(default=3, metadata={"help": "GCN order."})

    in_channels: int = field(default=1, metadata={"help": "Input channels."})
    out_channels: int = field(default=1, metadata={"help": "Output channels."})

    node_miss_rate: float = field(default=0.1, metadata={"help": "Kept for compatibility."})
    T_miss_len: int = field(default=12, metadata={"help": "Kept for compatibility."})

    lpls: Optional[List[List[float]]] = field(default=None, metadata={"help": "Optional precomputed Laplacian PE [N,32]."})
    adj: Optional[List[List[float]]] = field(default=None, metadata={"help": "Optional adjacency for building Laplacian PE."})
