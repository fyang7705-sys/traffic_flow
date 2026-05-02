from dataclasses import dataclass, field
from typing import List, Optional

from basicts.configs import BasicTSModelConfig


@dataclass
class STGCNConfig(BasicTSModelConfig):
    """Config class for a minimal STGCN-style model.

    Notes:
        - This implementation uses a simple spectral graph convolution with a
          precomputed (normalized) adjacency matrix and 1D temporal convolutions.
        - It is designed to work with BasicTS forecasting pipeline inputs of
          shape [B, T, N] or [B, T, N, C].
    """

    input_len: Optional[int] = field(default=None, metadata={"help": "Input sequence length."})
    output_len: Optional[int] = field(default=None, metadata={"help": "Output sequence length."})

    num_nodes: Optional[int] = field(default=None, metadata={"help": "Number of graph nodes."})
    in_channels: int = field(default=1, metadata={"help": "Input feature channels per node."})
    out_channels: int = field(default=1, metadata={"help": "Output feature channels per node."})

    hidden_channels: int = field(default=64, metadata={"help": "Hidden channels."})
    num_layers: int = field(default=2, metadata={"help": "Number of spatiotemporal blocks."})

    kernel_size: int = field(default=3, metadata={"help": "Temporal convolution kernel size."})
    dropout: float = field(default=0.0, metadata={"help": "Dropout probability."})

    # adjacency / normalization
    adj: Optional[List[List[float]]] = field(default=None, metadata={"help": "Adjacency matrix [N, N]."})
    add_self_loops: bool = field(default=True, metadata={"help": "Whether to add self-loops before normalization."})
    adj_normalization: str = field(default="sym", metadata={"help": "Adjacency normalization: 'sym' or 'rw'."})

    # output mapping
    use_projection: bool = field(default=True, metadata={"help": "Whether to use linear projection to output."})
    mlp_hidden: Optional[int] = field(default=None, metadata={"help": "Optional MLP hidden size for output head."})
