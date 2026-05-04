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
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.pe: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], :]


class PriorGuidedGraph(nn.Module):
    """
    Cleaner version:
    - single fusion (feature-level)
    - single prior injection (log-bias)
    """

    def __init__(self, embed_dim: int, bias_scale: float, attn_tau: float):
        super().__init__()
        self.embed_dim = embed_dim
        self.bias_scale = bias_scale
        self.attn_tau = attn_tau

        # dynamic
        self.q_dyn = nn.Linear(embed_dim, embed_dim)
        self.k_dyn = nn.Linear(embed_dim, embed_dim)

        # prior
        self.q_pri = nn.Linear(embed_dim, embed_dim)
        self.k_pri = nn.Linear(embed_dim, embed_dim)

        # fusion（关键）
        self.fuse = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def _normalize(self, A):
        return A / (A.sum(dim=-1, keepdim=True) + 1e-6)

    def forward(self, z, A_prior, node_emb):
        """
        z: [B,N,D]
        A_prior: [N,N]
        node_emb: [N,D]
        """
        B, N, D = z.shape
        scale = D ** 0.5

        A_norm = self._normalize(A_prior)

        # ===== dynamic branch =====
        qd = self.q_dyn(z)
        kd = self.k_dyn(z)

        # ===== prior branch（结构编码）=====
        context = A_norm @ node_emb   # [N,D]
        qp = self.q_pri(context).unsqueeze(0).expand(B, -1, -1)
        kp = self.k_pri(context).unsqueeze(0).expand(B, -1, -1)

        # ===== 融合（核心）=====
        q = self.fuse(torch.cat([qd, qp], dim=-1))
        k = self.fuse(torch.cat([kd, kp], dim=-1))

        # ===== attention =====
        score = torch.matmul(q, k.transpose(1, 2)) / scale

        # ===== 只保留一个 prior 注入（最关键）=====
        score = score + self.bias_scale * torch.log(A_norm.unsqueeze(0) + 1e-6)

        # ===== temperature =====
        score = score / self.attn_tau

        A = torch.softmax(score, dim=-1)

        return A

class TransformerGraphPropagator(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.drop = nn.Dropout(dropout)

        # graph propagation linear maps (equivalent to GCN weight W per layer)
        self.gcn_linears = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)])
        self.gcn_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])

    def forward(self, h: torch.Tensor, a_corr: torch.Tensor) -> torch.Tensor:
        # h: [B,N,D], a_corr: [B,N,N]
        out = h
        for layer, norm, gcn_lin, gcn_norm in zip(self.layers, self.norms, self.gcn_linears, self.gcn_norms):
            # (1) corrected-graph message passing + learnable projection (GCN-style)
            msg = torch.bmm(a_corr, out)          # [B,N,D]
            upd = torch.relu(gcn_lin(msg))        # [B,N,D]
            out = gcn_norm(out + self.drop(upd))  # residual

            # (2) transformer refinement on node tokens
            trans = layer(out)                    # [B,N,D]
            out = norm(out + self.drop(trans))
        return out


class BAFGNN(nn.Module):
    """BAFGNN.

    Differences from HimNet:
    - Remove temporal heterogeneity extraction branch (no tod/dow branch).
    - Replace GRU-based recurrence with transformer graph propagator.
    - Build dynamic graph via cross-attention fused with static prior adjacency.
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
        self.iter_refine_steps = int(config.iter_refine_steps)
        self.input_len = int(config.input_len)
        self.output_len = int(config.output_len)
        self.num_nodes = int(config.num_nodes)
        self.in_channels = int(config.in_channels)
        self.out_channels = int(config.out_channels)
        self.embed_dim = int(config.embed_dim)

        if config.adj is None:
            adj_tensor = torch.eye(self.num_nodes, dtype=torch.float32)
        else:
            adj_tensor = torch.as_tensor(config.adj, dtype=torch.float32)
            if adj_tensor.dim() != 2 or adj_tensor.shape[0] != self.num_nodes or adj_tensor.shape[1] != self.num_nodes:
                raise ValueError(
                    f"adj must be [num_nodes, num_nodes], got {tuple(adj_tensor.shape)} with num_nodes={self.num_nodes}"
                )
        self.register_buffer("a_prior", adj_tensor, persistent=True)

        # normalized prior adjacency for stable message passing in encoder
        a_row_sum = adj_tensor.sum(dim=-1, keepdim=True) + 1e-6
        self.register_buffer("a_prior_norm", adj_tensor / a_row_sum, persistent=True)

        self.node_emb = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim))

        # single temporal encoder branch (without temporal heterogeneity branch)
        self.input_proj = nn.Linear(self.in_channels, self.embed_dim)
        self.pos_enc = PositionalEncoding(self.embed_dim, max_len=max(2048, self.input_len + self.output_len))
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=int(config.num_heads),
                dim_feedforward=int(config.ff_dim),
                dropout=float(config.dropout),
                batch_first=True,
                activation="gelu",
            ),
            num_layers=int(config.num_layers),
        )

        # ----- spatial GCN stack applied at each time step (fixed prior adjacency) -----
        self.gcn_layers = int(getattr(config, "encoder_gcn_layers", 2))
        gcn_drop_cfg = getattr(config, "encoder_gcn_dropout", None)
        gcn_drop = float(config.dropout if gcn_drop_cfg is None else gcn_drop_cfg)
        self.encoder_gcn_drop = nn.Dropout(gcn_drop)
        # simple MLP-style GCN: h <- LN(h + Dropout(Linear(A@h))) repeated L times
        self.encoder_gcn_linears = nn.ModuleList(
            [nn.Linear(self.embed_dim, self.embed_dim) for _ in range(self.gcn_layers)]
        )
        self.encoder_gcn_norms = nn.ModuleList(
            [nn.LayerNorm(self.embed_dim) for _ in range(self.gcn_layers)]
        )

        # temporal readout / projection head (attention pooling over time)
        self.temporal_pool_q = nn.Parameter(torch.randn(self.embed_dim))  # [D]
        self.temporal_readout = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.temporal_readout_norm = nn.LayerNorm(self.embed_dim)

        self.graph_builder = PriorGuidedGraph(
            embed_dim=self.embed_dim,
            bias_scale=float(config.bias_scale),
            attn_tau=float(config.attn_tau),
        )

        graph_layers = int(getattr(config, "graph_transformer_layers", 2))
        self.graph_propagator = TransformerGraphPropagator(
            embed_dim=self.embed_dim,
            num_heads=int(config.num_heads),
            ff_dim=int(config.ff_dim),
            num_layers=graph_layers,
            dropout=float(config.dropout),
        )

        self.film = nn.Sequential(
            nn.Linear(self.embed_dim, int(config.film_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(config.film_hidden_dim), self.embed_dim * 2),
        )
        self.output_head = nn.Linear(self.embed_dim, self.output_len * self.out_channels)

    def _temporal_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence with per-step GCN over fixed adjacency then temporal Transformer.

        Pipeline (HimNet-style):
        1) Project inputs to D.
        2) For each time step, run multiple GCN layers using given adjacency (a_prior_norm).
        3) For each node, run a temporal Transformer across T to integrate information.
        4) Attention pooling over time to get [B,N,D].
        """
        # x: [B,T,N,C] -> [B,N,D]
        x = x[..., : self.in_channels]
        h0 = self.input_proj(x)  # [B,T,N,D]
        b, t, n, d = h0.shape

        a = torch.as_tensor(self.a_prior_norm, device=h0.device, dtype=h0.dtype).contiguous()  # [N,N]

        # (1) spatial encoding: per time-step multi-layer GCN
        h_list = []
        for ti in range(t):
            ht = h0[:, ti, :, :]  # [B,N,D]
            for lin, norm in zip(self.encoder_gcn_linears, self.encoder_gcn_norms):
                msg = torch.matmul(a, ht)              # [B,N,D]
                upd = lin(msg)
                upd = torch.relu(upd)
                ht = norm(ht + self.encoder_gcn_drop(upd))
            h_list.append(ht)
        h_sp = torch.stack(h_list, dim=1)  # [B,T,N,D]

        # (2) temporal integration per node via transformer
        h = h_sp.permute(0, 2, 1, 3).reshape(b * n, t, d)  # [B*N,T,D]
        h = self.pos_enc(h)
        h = self.temporal_encoder(h)  # [B*N,T,D]
        h = h.view(b, n, t, d).permute(0, 2, 1, 3).contiguous()  # [B,T,N,D]

        # (3) attention pooling over time steps
        score = torch.einsum("btnd,d->btn", h, self.temporal_pool_q) / math.sqrt(d)
        attn = torch.softmax(score, dim=1).unsqueeze(-1)  # [B,T,N,1]
        pooled = torch.sum(attn * h, dim=1)  # [B,N,D]

        z = self.temporal_readout_norm(pooled + self.temporal_readout(pooled))
        return z

    def forward(self, inputs: torch.Tensor, inputs_timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
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

        z = self._temporal_encode(x)  # [B,N,D]

        # ----- iterative graph refinement -----
        refine_steps = self.iter_refine_steps
        if refine_steps < 1:
            refine_steps = 1

        h = z
        for _ in range(refine_steps):
            a_corr = self.graph_builder(h, self.a_prior, self.node_emb)  # [B,N,N]
            h = self.graph_propagator(h, a_corr)  # [B,N,D]

        # keep lightweight node-wise heterogeneity modulation
        node_context = self.node_emb.unsqueeze(0).expand(h.shape[0], -1, -1)
        gamma, beta = torch.chunk(self.film(node_context), chunks=2, dim=-1)
        h = gamma * h + beta

        y = self.output_head(h)  # [B,N,T_out*C_out]
        y = y.view(y.shape[0], self.num_nodes, self.output_len, self.out_channels)
        y = y.permute(0, 2, 1, 3).contiguous()
        if self.out_channels == 1:
            y = y.squeeze(-1)
        return y
