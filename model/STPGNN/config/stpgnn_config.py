from dataclasses import dataclass, field
from typing import Optional

from basicts.configs import BasicTSModelConfig


@dataclass
class STPGNNConfig(BasicTSModelConfig):
    """Config class for STPGNN model."""

    input_len: Optional[int] = field(default=None, metadata={"help": "Input sequence length."})
    output_len: Optional[int] = field(default=None, metadata={"help": "Output sequence length."})
    num_nodes: Optional[int] = field(default=None, metadata={"help": "Number of nodes/features."})

    in_dim: int = field(default=1, metadata={"help": "Input channels per node."})
    dropout: float = field(default=0.1, metadata={"help": "Dropout rate."})
    topk: int = field(default=35, metadata={"help": "Top-k nodes for pivotal graph."})

    residual_channels: int = field(default=32, metadata={"help": "Residual channels."})
    dilation_channels: int = field(default=32, metadata={"help": "Dilation channels."})
    end_channels: int = field(default=512, metadata={"help": "Channels before output layer."})

    kernel_size: int = field(default=2, metadata={"help": "Temporal kernel size."})
    blocks: int = field(default=4, metadata={"help": "Number of blocks."})
    layers: int = field(default=2, metadata={"help": "Layers per block."})

    days: int = field(default=48, metadata={"help": "Temporal slots used by STPGNN."})
    time_of_day_size: int = field(default=288, metadata={"help": "Timestamps per day."})

    dims: int = field(default=32, metadata={"help": "Embedding dimension."})
    order: int = field(default=2, metadata={"help": "Graph convolution order."})
    normalization: str = field(default="batch", metadata={"help": "Normalization type: batch or layer."})
