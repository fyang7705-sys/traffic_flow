from dataclasses import dataclass, field
from typing import List, Optional

from basicts.configs import BasicTSModelConfig


@dataclass
class DFDGCNConfig(BasicTSModelConfig):
    input_len: Optional[int] = field(default=None)
    output_len: Optional[int] = field(default=None)
    num_nodes: Optional[int] = field(default=None)
    in_dim: int = field(default=3)
    residual_channels: int = field(default=32)
    dilation_channels: int = field(default=32)
    skip_channels: int = field(default=256)
    end_channels: int = field(default=512)
    kernel_size: int = field(default=2)
    blocks: int = field(default=4)
    layers: int = field(default=2)
    dropout: float = field(default=0.3)
    fft_emb: int = field(default=32)
    subgraph: int = field(default=20)
    identity_emb: int = field(default=32)
    hidden_emb: int = field(default=64)
    affine: bool = field(default=True)
    gcn_bool: bool = field(default=True)
    addaptadj: bool = field(default=True)
    a: float = field(default=10.0)
    time_of_day_size: int = field(default=288)
    day_of_week_size: int = field(default=7)
    adj: Optional[List[List[float]]] = field(default=None)
