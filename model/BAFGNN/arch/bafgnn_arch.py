import math
from typing import Optional

import torch
from torch import nn

from ..config.bafgnn_config import BAFGNNConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, T, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], :]


class BAFGNN(nn.Module):
    """BasicTS-compatible BAFGNN.

    Pipeline:
      1) iTransformer-lite temporal encoding -> Z [B, N, D]
      2) Graph-biased attention with static A_prior
      3) FiLM modulation by node-id embedding
      4) Linear prediction head to multi-step forecast
    """

    def __init__(self, config: BAFGNNConfig):
        super().__init__()
        if config.input_len is None or config.output_len is None or config.num_nodes is None:
            raise ValueError("BAFGNNConfig.input_len/output_len/num_nodes must be set")
        if config.embed_dim % config.num_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads, got {config.embed_dim} and {config.num_heads}"
            )
        if config.attn_tau <= 0:
            raise ValueError(f"attn_tau must be > 0, got {config.attn_tau}")

        self.input_len = int(config.input_len)
        self.output_len = int(config.output_len)
        self.num_nodes = int(config.num_nodes)
        self.in_channels = int(config.in_channels)
        self.out_channels = int(config.out_channels)
        self.embed_dim = int(config.embed_dim)
        self.bias_scale = float(config.bias_scale)
        self.attn_tau = float(config.attn_tau)

        if config.adj is None:
            adj_tensor = torch.eye(self.num_nodes, dtype=torch.float32)
        else:
            adj_tensor = torch.as_tensor(config.adj, dtype=torch.float32)
            if adj_tensor.dim() != 2 or adj_tensor.shape[0] != self.num_nodes or adj_tensor.shape[1] != self.num_nodes:
                raise ValueError(
                    f"adj must be [num_nodes, num_nodes], got {tuple(adj_tensor.shape)} with num_nodes={self.num_nodes}"
                )
        self.register_buffer("a_prior", adj_tensor, persistent=True)

        self.node_emb = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim))
        self.input_proj = nn.Linear(self.in_channels, self.embed_dim)
        self.struct_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.pos_enc = PositionalEncoding(self.embed_dim, max_len=max(2048, self.input_len + self.output_len))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=int(config.num_heads),
            dim_feedforward=int(config.ff_dim),
            dropout=float(config.dropout),
            batch_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(enc_layer, num_layers=int(config.num_layers))
        self.dropout = nn.Dropout(float(config.dropout))

        self.q_proj = nn.Linear(self.embed_dim * 3, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim * 3, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.film = nn.Sequential(
            nn.Linear(self.embed_dim, int(config.film_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(config.film_hidden_dim), self.embed_dim * 2),
        )

        self.head = nn.Linear(self.embed_dim, self.output_len * self.out_channels)

    def _temporal_encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, C] -> Z: [B, N, D]
        x = x[..., : self.in_channels]
        x = self.input_proj(x)  # [B, T, N, D]
        x = x + self.node_emb.unsqueeze(0).unsqueeze(0)
        b, t, n, d = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b * n, t, d)  # [B*N, T, D]
        x = self.pos_enc(x)
        x = self.dropout(self.temporal_encoder(x))
        z = x[:, -1, :].reshape(b, n, d)  # take last token as node representation
        return z

    def forward(self, inputs: torch.Tensor, inputs_timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = inputs.shape[0]
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

        z = self._temporal_encode(x)  # [B, N, D]

        node_context = self.node_emb.unsqueeze(0).expand(z.shape[0], -1, -1)  # [B, N, D]
        A = self.a_prior
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)   # row normalize
        struct_context = torch.matmul(A, self.node_emb)
        struct_context = struct_context.unsqueeze(0).expand(B, -1, -1)
        struct_context = self.struct_proj(struct_context)  # [B, N, D]
        q_in = torch.cat([z, node_context, struct_context], dim=-1)
        k_in = torch.cat([z, node_context, struct_context], dim=-1)

        q = self.q_proj(q_in)  # [B, N, D]
        k = self.k_proj(k_in)  # [B, N, D]
        v = self.v_proj(z)     # [B, N, D]

        scale = math.sqrt(self.embed_dim)
        score = torch.matmul(q, k.transpose(1, 2)) / scale
        # score = data_score + self.bias_scale * self.a_prior.unsqueeze(0)
        alpha = torch.softmax(score / self.attn_tau, dim=-1)  # [B, N, N]
        h = torch.matmul(alpha, v)  # [B, N, D]

        film_params = self.film(node_context)  # [B, N, 2D]
        gamma, beta = torch.chunk(film_params, chunks=2, dim=-1)
        h = gamma * h + beta

        y = self.head(h)  # [B, N, T_out * C_out]
        y = y.view(y.shape[0], self.num_nodes, self.output_len, self.out_channels)
        y = y.permute(0, 2, 1, 3).contiguous()  # [B, T_out, N, C_out]
        if self.out_channels == 1:
            y = y.squeeze(-1)
        return y
