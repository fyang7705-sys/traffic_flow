from dataclasses import dataclass, field
from typing import List, Optional

from basicts.configs import BasicTSModelConfig


@dataclass
class BAFGNNConfig(BasicTSModelConfig):
    # ----- data / shapes -----
    input_len: Optional[int] = field(default=None, metadata={"help": "Input sequence length T_in."})
    output_len: Optional[int] = field(default=None, metadata={"help": "Output sequence length T_out."})
    num_nodes: Optional[int] = field(default=None, metadata={"help": "Number of nodes N."})

    in_channels: int = field(default=1, metadata={"help": "Input feature channels per node C_in."})
    out_channels: int = field(default=1, metadata={"help": "Output feature channels per node C_out."})

    # ----- model width -----
    embed_dim: int = field(default=64, metadata={"help": "Hidden embedding dimension D."})

    # ----- temporal transformer (node-wise) -----
    num_heads: int = field(default=4, metadata={"help": "Attention heads for temporal Transformer."})
    num_layers: int = field(default=2, metadata={"help": "Number of temporal Transformer encoder layers."})
    ff_dim: int = field(default=256, metadata={"help": "Feed-forward hidden dimension in Transformer layers."})
    dropout: float = field(default=0.1, metadata={"help": "Dropout rate used in Transformer/readout."})

    # ----- fixed prior adjacency (encoder) -----
    adj: Optional[List[List[float]]] = field(default=None, metadata={"help": "Static prior adjacency matrix [N,N]."})
    encoder_gcn_layers: int = field(default=2, metadata={"help": "#GCN layers applied at each time step in encoder (using fixed adj)."})
    encoder_gcn_dropout: Optional[float] = field(
        default=None,
        metadata={"help": "Dropout used in encoder GCN stack. If None, fallback to dropout."},
    )

    # ----- dynamic graph builder (prior + data) -----
    bias_scale: float = field(default=1.0, metadata={"help": "Scale factor for prior bias terms in graph builder."})
    attn_tau: float = field(default=1.0, metadata={"help": "Softmax temperature for graph-builder attention (>0)."})

    # ----- graph propagator (after graph_builder) -----
    graph_transformer_layers: int = field(default=2, metadata={"help": "#layers in graph propagator Transformer."})

    # ----- readout modulation -----
    film_hidden_dim: int = field(default=64, metadata={"help": "Hidden size of FiLM MLP for node-wise modulation."})

    # ----- iterative graph refinement (graph_builder <-> graph_propagator) -----
    iter_refine_steps: int = field(
        default=1,
        metadata={"help": "#iterations of graph refinement: rebuild corrected graph then propagate repeatedly."},
    )
