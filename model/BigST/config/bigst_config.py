from dataclasses import dataclass, field
from typing import List, Optional

from basicts.configs import BasicTSModelConfig


@dataclass
class BigSTConfig(BasicTSModelConfig):
    input_len: Optional[int] = field(default=None)
    output_len: Optional[int] = field(default=None)
    num_nodes: Optional[int] = field(default=None)
    in_dim: int = field(default=3)
    hid_dim: int = field(default=32)
    random_feature_dim: int = field(default=64)
    node_emb_dim: int = field(default=32)
    time_emb_dim: int = field(default=32)
    tau: float = field(default=0.25)
    dropout: float = field(default=0.3)
    use_residual: bool = field(default=True)
    use_bn: bool = field(default=True)
    use_spatial: bool = field(default=True)
    use_long: bool = field(default=False)
    use_time_embedding: bool = field(default=True)
    time_of_day_size: int = field(default=288)
    day_of_week_size: int = field(default=7)
    adj: Optional[List[List[float]]] = field(default=None)
