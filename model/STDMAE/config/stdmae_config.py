from dataclasses import dataclass, field
from typing import List, Optional

from basicts.configs import BasicTSModelConfig


@dataclass
class STDMAEConfig(BasicTSModelConfig):
    input_len: Optional[int] = field(default=None)
    output_len: Optional[int] = field(default=None)
    num_nodes: Optional[int] = field(default=None)
    patch_size: int = field(default=12)
    mask_embed_dim: int = field(default=96)
    mask_num_heads: int = field(default=4)
    mask_mlp_ratio: int = field(default=4)
    mask_dropout: float = field(default=0.1)
    mask_ratio: float = field(default=0.25)
    encoder_depth: int = field(default=4)
    decoder_depth: int = field(default=1)
    gw_residual_channels: int = field(default=32)
    gw_dilation_channels: int = field(default=32)
    gw_skip_channels: int = field(default=256)
    gw_end_channels: int = field(default=512)
    gw_blocks: int = field(default=4)
    gw_layers: int = field(default=2)
    gw_dropout: float = field(default=0.3)
    adj: Optional[List[List[float]]] = field(default=None)
