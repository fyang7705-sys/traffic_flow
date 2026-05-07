import torch
from torch import nn


class TimeEmbedding(nn.Module):
    """
    节点内时间建模使用 GRU，不使用 Transformer。

    输出:
        g_global: (B, output_dim)     全局 regime/time embedding
        e_node:   (B, N, output_dim)  节点级 time embedding
    """

    def __init__(
        self,
        in_steps,
        num_nodes,
        input_dim,
        output_dim,
        node_embedding_dim,
        d_model=64,
        nhead=4,
        num_regimes=4,
        dropout=0.1,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")

        self.in_steps = in_steps
        self.num_nodes = num_nodes
        self.d_model = d_model

        # 显式动力学描述符投影: [load_t, |Δ|_t, entropy_t]
        self.dyn_proj = nn.Linear(3, d_model)

        # 每个节点每个时间步的 raw feature 投影
        self.value_proj = nn.Linear(input_dim, d_model)


        # 节点身份注入
        self.node_id_proj = nn.Linear(node_embedding_dim, d_model)

        # 节点内时间 GRU: 每个节点独立处理自己的 T 步历史
        self.time_gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
        )

        self.time_norm = nn.LayerNorm(d_model)
        self.time_dropout = nn.Dropout(dropout)

        # regime queries: 全局态势 prototype
        self.regime_queries = nn.Parameter(
            torch.randn(num_regimes, d_model) * 0.02
        )

        self.global_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.regime_gate = nn.Linear(d_model, 1)

        # 全局输出头
        self.out_global = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )


    @staticmethod
    def _dynamics_descriptors(x):
        # x: (B, T, N, C)
        B, T, N, C = x.shape

        # 整体负载 / 活跃度
        load = x.abs().mean(dim=(2, 3))  # (B, T)

        # 系统变化幅度，建议除以 sqrt(N*C)，避免节点数变大导致尺度过大
        diff = torch.zeros_like(load)
        diff[:, 1:] = (
            (x[:, 1:] - x[:, :-1])
            .flatten(2)
            .norm(dim=-1)
            / ((N * C) ** 0.5)
        )

        # 空间分布熵
        p = torch.softmax(x.mean(dim=-1), dim=-1)  # (B, T, N)
        ent = -(p * p.clamp_min(1e-9).log()).sum(dim=-1)  # (B, T)

        # 归一化到大致 [0, 1]
        ent = ent / torch.log(
            torch.tensor(float(N), device=x.device, dtype=x.dtype)
        ).clamp_min(1e-6)

        desc = torch.stack([load, diff, ent], dim=-1)  # (B, T, 3)
        return desc

    def forward(self, x, node_embedding):
        """
        x:
            (B, T, N, C)

        node_embedding:
            (N, node_embedding_dim)
        """
        B, T, N, C = x.shape

        if T != self.in_steps:
            raise ValueError(f"Expected T={self.in_steps}, but got T={T}")

        if N != self.num_nodes:
            raise ValueError(f"Expected N={self.num_nodes}, but got N={N}")

        # ------------------------------------------------------------
        # 1) 节点内 GRU 时间建模: (B*N, T, d)
        # ------------------------------------------------------------
        v = self.value_proj(x)  # B, T, N, d
        v = v.permute(0, 2, 1, 3).contiguous().view(B * N, T, self.d_model)

        node_id = self.node_id_proj(node_embedding)  # N, d
        node_id = (
            node_id.unsqueeze(0)
            .expand(B, -1, -1)
            .contiguous()
            .view(B * N, 1, self.d_model)
        )

        v = v + node_id
        h, _ = self.time_gru(v)
        h = self.time_norm(h)
        h = self.time_dropout(h)

        h = h.view(B, N, T, self.d_model)  # (B, N, T, d)

        # ------------------------------------------------------------
        # 2) 显式动力学描述符注入到全局时间通道
        # ------------------------------------------------------------
        desc = self._dynamics_descriptors(x)  # (B, T, 3)
        desc = self.dyn_proj(desc)  # (B, T, d)

        global_tokens = h.mean(dim=1) + desc  # (B, T, d)

        # ------------------------------------------------------------
        # 3) regime queries 池化全局态势
        # ------------------------------------------------------------
        Q = self.regime_queries.unsqueeze(0).expand(B, -1, -1)  # (B, K, d)

        regimes, _ = self.global_attn(
            Q,
            global_tokens,
            global_tokens,
        )  # (B, K, d)

        w = torch.softmax(self.regime_gate(regimes), dim=1)  # (B, K, 1)

        g = (w * regimes).sum(dim=1)  # (B, d)

        # ------------------------------------------------------------
        # 4) 输出全局 embedding 和节点级 embedding
        # ------------------------------------------------------------
        g_global = self.out_global(g)  # (B, output_dim)


        return g_global
