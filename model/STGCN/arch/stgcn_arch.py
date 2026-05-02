from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from ..config.stgcn_config import STGCNConfig


def _normalize_adj(adj: torch.Tensor, add_self_loops: bool = True, kind: str = "sym") -> torch.Tensor:
    """Normalize adjacency.

    Args:
        adj: [N, N]
        add_self_loops: whether add I before norm
        kind: 'sym' (D^-1/2 A D^-1/2) or 'rw' (D^-1 A)
    """
    if adj.dim() != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adj must be [N,N], got {tuple(adj.shape)}")

    a = adj
    if add_self_loops:
        a = a + torch.eye(a.shape[0], device=a.device, dtype=a.dtype)

    deg = a.sum(dim=1)  # [N]
    if kind == "rw":
        deg_inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
        d_inv = torch.diag(deg_inv)
        return d_inv @ a

    if kind == "sym":
        deg_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
        d_inv_sqrt = torch.diag(deg_inv_sqrt)
        return d_inv_sqrt @ a @ d_inv_sqrt

    raise ValueError(f"Unsupported adj_normalization: {kind}")


class GraphConv(nn.Module):
    """A simple graph convolution: X' = A_hat X W."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        """Args:
            x: [B, T, N, C]
            a_hat: [N, N]
        Returns:
            [B, T, N, C']
        """
        # aggregate neighbors: [B, T, N, C]
        x_agg = torch.einsum("nm,btnc->btmc", a_hat, x)
        return self.proj(x_agg)


class STBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.temp1 = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), padding=(padding, 0))
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.temp2 = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), padding=(padding, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, N]
        h = self.temp1(x)
        h = self.act(h)
        h = self.drop(h)
        h = self.temp2(h)
        return self.act(h + x)


class STGCN(nn.Module):
    """A minimal STGCN-like forecaster.

    Input:
        inputs: [B, T, N] or [B, T, N, C]

    Output:
        prediction: [B, output_len, N] or [B, output_len, N, out_channels]

    Notes:
        - Adjacency can be provided at init time via config/argument.
        - For BasicTS compatibility, forward signature accepts (inputs, inputs_timestamps)
          but timestamps are unused.
    """

    def __init__(
        self,
        config: STGCNConfig,
        adj: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        if config.input_len is None or config.output_len is None:
            raise ValueError("STGCNConfig.input_len/output_len must be set")
        if config.num_nodes is None:
            raise ValueError("STGCNConfig.num_nodes must be set")

        self.input_len = int(config.input_len)
        self.output_len = int(config.output_len)
        self.num_nodes = int(config.num_nodes)
        self.in_channels = int(config.in_channels)
        self.out_channels = int(config.out_channels)

        # adjacency buffer (normalized)
        if adj is None:
            adj = torch.eye(self.num_nodes)
        a_hat = _normalize_adj(adj.float(), add_self_loops=config.add_self_loops, kind=config.adj_normalization)
        self.register_buffer("a_hat", a_hat)

        hidden = int(config.hidden_channels)

        self.input_proj = nn.Linear(self.in_channels, hidden)
        self.gconv_in = GraphConv(hidden, hidden)

        self.st_blocks = nn.ModuleList([STBlock(hidden, config.kernel_size, config.dropout) for _ in range(config.num_layers)])

        self.gconv_out = GraphConv(hidden, hidden)

        # map from T_in -> T_out per node/channel
        self.temporal_out = nn.Linear(self.input_len, self.output_len)

        if config.use_projection:
            if config.mlp_hidden is not None:
                self.head = nn.Sequential(
                    nn.Linear(hidden, int(config.mlp_hidden)),
                    nn.ReLU(),
                    nn.Linear(int(config.mlp_hidden), self.out_channels),
                )
            else:
                self.head = nn.Linear(hidden, self.out_channels)
        else:
            self.head = nn.Identity()

    def forward(self, inputs: torch.Tensor, inputs_timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Accept [B, T, N] -> [B, T, N, 1]
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

        # channel embedding
        x = self.input_proj(x)  # [B, T, N, hidden]

        # spatial conv
        x = self.gconv_in(x, self.a_hat)  # [B, T, N, hidden]

        # temporal blocks operate on [B, C, T, N]
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, hidden, T, N]
        for blk in self.st_blocks:
            x = blk(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, T, N, hidden]

        # spatial conv
        x = self.gconv_out(x, self.a_hat)  # [B, T, N, hidden]

        # temporal mapping to output_len: apply Linear over time dim
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, N, hidden, T]
        x = self.temporal_out(x)  # [B, N, hidden, T_out]
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, T_out, N, hidden]

        y = self.head(x)  # [B, T_out, N, out_channels]
        if self.out_channels == 1:
            y = y.squeeze(-1)  # [B, T_out, N]
        return y
