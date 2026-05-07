from torch import nn
import torch
import math


class CrossAttentionGraphFusion(nn.Module):
    """
    利用静态邻接图和 node_embedding 生成的自适应图进行动态融合。

    输入:
        node_embedding:   N, E
        adaptive_support: N, N  (由 node_embedding 派生)
        static_support:   N, N  (物理结构先验)

    输出:
        fused_support:    N, N

    设计要点 (相对原版的差异):

    1. cross_support 用 static_support 作 attention bias,
       让 cross_support 成为 "static 骨架上的语义精化版",
       而不是另一份 node_embedding 的自相似图。
       这避免了 cross 与 adaptive 在语义上的冗余。

    2. 多头注意力先在 head 维度 mean 收敛成单通道 cross_support,
       避免 H 个头各自占据 candidates 通道、稀释 static_support
       在 softmax 中的份额。

    3. 三通道融合 (adaptive / static / cross),并给 static 通道
       一个可学习的、初始为正的先验偏置,确保训练初期 static
       占主导,后续再由数据驱动调整。
    """

    def __init__(
        self,
        num_nodes,
        node_embedding_dim,
        attn_dim=None,
        nhead: int = 4,
        fusion_hidden_dim=16,
        dropout=0.1,
        static_bias_init: float = 2.0,    # static 通道初始先验 logit
        static_attn_bias: float = 1.0,    # cross_support attention 中的 static 偏置强度
    ):
        super().__init__()

        self.num_nodes = num_nodes

        if attn_dim is None:
            attn_dim = node_embedding_dim

        if attn_dim % nhead != 0:
            raise ValueError(
                f"attn_dim={attn_dim} must be divisible by nhead={nhead}"
            )

        self.attn_dim = int(attn_dim)
        self.nhead = int(nhead)
        self.head_dim = self.attn_dim // self.nhead

        self.q_proj = nn.Linear(node_embedding_dim, self.attn_dim)
        self.k_proj = nn.Linear(node_embedding_dim, self.attn_dim)
        self.dropout = nn.Dropout(dropout)

        # cross_attention 中, static_support 作为加性 attention bias
        # logit += static_attn_bias * 1[static > 0]
        # 软偏置: static 上有边的位置 attention 更强, 但不硬截断其他位置
        self.static_attn_bias = float(static_attn_bias)

        # 三通道边级融合 MLP (adaptive, static, cross)
        self.edge_fusion = nn.Sequential(
            nn.Linear(3, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, 3),
        )

        # 给 static 通道一个可学习的先验 logit (初始为正,优先 static)
        # shape: (3,) -> [adaptive, static, cross]
        prior = torch.zeros(3)
        prior[1] = float(static_bias_init)
        self.fusion_prior = nn.Parameter(prior)

    def _row_normalize(self, adj, eps=1e-8):
        adj = torch.relu(adj)
        return adj / (adj.sum(dim=-1, keepdim=True) + eps)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """N, attn_dim -> nhead, N, head_dim"""
        # N, (H*D) -> N, H, D -> H, N, D
        return (
            x.view(x.shape[0], self.nhead, self.head_dim)
            .permute(1, 0, 2)
            .contiguous()
        )

    def forward(self, node_embedding, adaptive_support, static_support):
        """
        node_embedding:
            N, E

        adaptive_support:
            N, N

        static_support:
            N, N
        """

        static_support = static_support.to(
            device=adaptive_support.device,
            dtype=adaptive_support.dtype,
        )

        # 静态图先做行归一化, 避免尺度与 adaptive 不一致
        static_support = self._row_normalize(static_support)

        # 边存在性掩码 (归一化后的 static, 非零等价于原始非零)
        # N, N
        static_edge_mask = (static_support > 0).to(static_support.dtype)

        # ------------------------------------------------------------------
        # 1. 用静态图聚合 node_embedding, 得到静态结构感知的节点表示
        # ------------------------------------------------------------------
        # N, E
        static_context = torch.matmul(static_support, node_embedding)

        # ------------------------------------------------------------------
        # 2. 多头交叉注意力, 以 static_support 作 attention bias
        #    让 cross_support 成为 "static 骨架上的语义精化"
        # ------------------------------------------------------------------
        Q = self.q_proj(node_embedding)        # N, attn_dim
        K = self.k_proj(static_context)        # N, attn_dim

        Qh = self._split_heads(Q)              # H, N, D
        Kh = self._split_heads(K)              # H, N, D

        # H, N, N
        attn_logits = torch.matmul(Qh, Kh.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # static 边上加成 (软掩码), 引导 cross 偏向 static 结构
        # 1, N, N -> 广播到 H, N, N
        attn_bias = (self.static_attn_bias * static_edge_mask).unsqueeze(0)
        attn_logits = attn_logits + attn_bias

        cross_support_h = torch.softmax(attn_logits, dim=-1)
        cross_support_h = self.dropout(cross_support_h)

        # 多头平均, 只保留单通道, 避免稀释 static 在融合 softmax 中的份额
        # H, N, N -> N, N
        cross_support = cross_support_h.mean(dim=0)

        # ------------------------------------------------------------------
        # 3. 三通道边级融合 (adaptive / static / cross)
        #    softmax 之前加可学习的先验偏置, 训练初期让 static 占主导
        # ------------------------------------------------------------------
        # N, N, 3
        candidates = torch.stack(
            [adaptive_support, static_support, cross_support],
            dim=-1,
        )

        # N, N, 3 + (3,) -> N, N, 3
        fusion_logits = self.edge_fusion(candidates) + self.fusion_prior
        fusion_weight = torch.softmax(fusion_logits, dim=-1)

        # N, N
        fused_support = (
            fusion_weight[..., 0] * adaptive_support
            + fusion_weight[..., 1] * static_support
            + fusion_weight[..., 2] * cross_support
        )

        fused_support = self._row_normalize(fused_support)
        return fused_support

