import math

import torch
from torch import nn

from ..config.pdformer_config import PDFormerConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, T, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], :]


class PDFormer(nn.Module):
    """BasicTS-compatible PDFormer-style encoder."""

    def __init__(self, config: PDFormerConfig):
        super().__init__()
        if config.input_len is None or config.output_len is None or config.num_nodes is None:
            raise ValueError("PDFormerConfig.input_len/output_len/num_nodes must be set")

        self.input_len = int(config.input_len)
        self.output_len = int(config.output_len)
        self.num_nodes = int(config.num_nodes)
        self.input_dim = int(config.input_dim)
        self.output_dim = int(config.output_dim)
        self.embed_dim = int(config.embed_dim)

        self.input_proj = nn.Linear(self.input_dim, self.embed_dim)
        self.node_emb = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim))
        self.pos_enc = PositionalEncoding(self.embed_dim, max_len=max(2048, self.input_len + self.output_len))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=int(config.num_heads),
            dim_feedforward=int(config.ff_dim),
            dropout=float(config.dropout),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(config.num_layers))
        self.dropout = nn.Dropout(float(config.dropout))
        self.time_proj = nn.Linear(self.input_len, self.output_len)
        self.out_proj = nn.Linear(self.embed_dim, self.output_dim)

    def forward(self, inputs: torch.Tensor, inputs_timestamps: torch.Tensor = None) -> torch.Tensor:
        if inputs.dim() == 3:
            x = inputs.unsqueeze(-1)
        elif inputs.dim() == 4:
            x = inputs
        else:
            raise ValueError(f"inputs must be [B,T,N] or [B,T,N,C], got {tuple(inputs.shape)}")

        if x.shape[1] != self.input_len:
            raise ValueError(f"Expected input_len={self.input_len}, got T={x.shape[1]}")
        if x.shape[2] != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got N={x.shape[2]}")

        x = x[..., : self.input_dim]  # [B,T,N,C]
        x = self.input_proj(x)  # [B,T,N,D]
        x = x + self.node_emb.unsqueeze(0).unsqueeze(0)

        b, t, n, d = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b * n, t, d)
        x = self.pos_enc(x)
        x = self.dropout(self.encoder(x))  # [B*N,T,D]

        x = x.transpose(1, 2)  # [B*N,D,T]
        x = self.time_proj(x)  # [B*N,D,T_out]
        x = x.transpose(1, 2)  # [B*N,T_out,D]
        x = self.out_proj(x)   # [B*N,T_out,C_out]
        x = x.reshape(b, n, self.output_len, self.output_dim).permute(0, 2, 1, 3).contiguous()
        if self.output_dim == 1:
            x = x.squeeze(-1)
        return x

