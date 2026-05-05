from torch import nn
import torch
import math


class CrossAttentionGraphFusion(nn.Module):
    """
    利用静态邻接图和 node_embedding 生成的自适应图进行动态融合。

    输入：
        node_embedding:   N, E
        adaptive_support: N, N
        static_support:   N, N

    输出：
        fused_support:    N, N
    """

    def __init__(
        self,
        num_nodes,
        node_embedding_dim,
        attn_dim=None,
        nhead: int = 4,
        long_pattern_dim=2,
        fusion_hidden_dim=16,
        dropout=0.1,
        head_fusion: str = "mean",
    ):
        super().__init__()

        self.num_nodes = num_nodes

        if attn_dim is None:
            attn_dim = node_embedding_dim

        if attn_dim % nhead != 0:
            raise ValueError(f"attn_dim={attn_dim} must be divisible by nhead={nhead}")

        self.attn_dim = int(attn_dim)
        self.nhead = int(nhead)
        self.head_dim = self.attn_dim // self.nhead
        self.head_fusion = head_fusion

        # 使用专门的 embedding 投影用于静态图语境（避免直接共用 node_embedding）
        self.long_pattern_encoder = nn.Sequential(
            nn.Linear(long_pattern_dim, node_embedding_dim),
            nn.ReLU(),
            nn.Linear(node_embedding_dim, node_embedding_dim)
        )

        self.q_proj = nn.Linear(node_embedding_dim, self.attn_dim)
        self.k_proj = nn.Linear(node_embedding_dim, self.attn_dim)
        self.dropout = nn.Dropout(dropout)

        # 对 adaptive / static / cross(heads) 做边级别动态融合
        self.edge_fusion = nn.Sequential(
            nn.Linear(2 + self.nhead, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, 2 + self.nhead),
        )

    def _row_normalize(self, adj, eps=1e-8):
        adj = torch.relu(adj)
        return adj / (adj.sum(dim=-1, keepdim=True) + eps)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """N, attn_dim -> nhead, N, head_dim"""
        # N, (H*D) -> N, H, D -> H, N, D
        return x.view(x.shape[0], self.nhead, self.head_dim).permute(1, 0, 2).contiguous()

    def forward(self, adaptive_support, static_support, node_embedding, long_pattern):
        """
        node_embedding:
            N, E

        adaptive_support:
            N, N

        static_support:
            N, N
        """


        # 静态图先做行归一化，避免尺度和 adaptive_support 不一致
        static_support = self._row_normalize(static_support)

        # -------------------------------------------------------
        # 1. 用静态图聚合“专门的 embedding”，得到静态结构感知的节点表示
        # -------------------------------------------------------
        static_node_embedding = self.long_pattern_encoder(long_pattern)
        static_context = torch.matmul(static_support, static_node_embedding)
    
        # -------------------------------------------------------
        # 2. 多头交叉注意力：
        #    query 来自 node_embedding
        #    key 来自 static_context
        # -------------------------------------------------------
        # N, attn_dim
        Q = self.q_proj(node_embedding)
        K = self.k_proj(static_context)

        # H, N, D
        Qh = self._split_heads(Q)
        Kh = self._split_heads(K)

        # H, N, N
        attn_logits = torch.matmul(Qh, Kh.transpose(-1, -2)) / math.sqrt(self.head_dim)
        cross_support_h = torch.softmax(attn_logits, dim=-1)
        cross_support_h = self.dropout(cross_support_h)

        # 1) cross_support 保留多头: H,N,N -> N,N,H
        cross_support = cross_support_h.permute(1, 2, 0).contiguous()

        # 2) candidates 扩展: N,N,(2+H)
        candidates = torch.cat(
            [
                adaptive_support.unsqueeze(-1),  # N,N,1
                static_support.unsqueeze(-1),    # N,N,1
                cross_support,                   # N,N,H
            ],
            dim=-1,
        )

        # 3) 融合权重: N,N,(2+H)
        fusion_logits = self.edge_fusion(candidates)
        fusion_weight = torch.softmax(fusion_logits, dim=-1)

        # N,N
        fused_support = (
            fusion_weight[..., 0] * adaptive_support
            + fusion_weight[..., 1] * static_support
            + (fusion_weight[..., 2:] * cross_support).sum(dim=-1)
        )

        fused_support = self._row_normalize(fused_support)
        return fused_support