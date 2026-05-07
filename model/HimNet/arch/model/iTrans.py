import torch
from torch import nn


class SimpleTimeEmbedding(nn.Module):
    """
    MLP-Mixer 风格的时间模式提取。

    核心:
      Block = TokenMixing(沿节点维) + ChannelMixing(沿特征维)
    比 self-attention 更简单, 监督信号更直接, 在小数据集上更稳。
    """

    def __init__(
        self,
        in_steps,
        num_nodes,
        input_dim,
        output_dim,
        node_embedding_dim,
        d_model=32,
        token_hidden=None,
        channel_hidden=None,
        dropout=0.1,
        use_shared_node_embedding=True,
    ):
        super().__init__()
        self.in_steps = in_steps
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.d_model = d_model
        self.use_shared_node_embedding = use_shared_node_embedding

        # 每节点 token 的特征 = T * C
        self.value_embedding = nn.Linear(in_steps * input_dim, d_model)

        if use_shared_node_embedding:
            self.node_embedding_proj = nn.Linear(node_embedding_dim, d_model)

        if token_hidden is None:
            token_hidden = max(num_nodes // 2, 32)
        if channel_hidden is None:
            channel_hidden = d_model * 2

        # token mixing: 沿节点维度的 MLP
        self.token_norm = nn.LayerNorm(d_model)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_nodes, token_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_hidden, num_nodes),
        )

        # channel mixing: 沿特征维度的 MLP
        self.channel_norm = nn.LayerNorm(d_model)
        self.channel_mlp = nn.Sequential(
            nn.Linear(d_model, channel_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_hidden, d_model),
        )

        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, output_dim)

    def forward(self, x, node_embedding=None):
        """
        x: B, T, N, C
        node_embedding: N, node_embedding_dim
        return: B, N, output_dim
        """
        B, T, N, C = x.shape

        # B, T, N, C -> B, N, T, C -> B, N, T*C
        x = x.permute(0, 2, 1, 3).contiguous().view(B, N, T * C)

        # B, N, T*C -> B, N, d_model
        h = self.value_embedding(x)

        # 节点身份
        if self.use_shared_node_embedding:
            id_emb = self.node_embedding_proj(node_embedding).unsqueeze(0)  # 1, N, d
            h = h + id_emb

        # token mixing (沿节点维)
        residual = h
        h = self.token_norm(h)
        h = h.transpose(1, 2)              # B, d, N
        h = self.token_mlp(h)              # B, d, N
        h = h.transpose(1, 2)              # B, N, d
        h = h + residual

        # channel mixing (沿特征维)
        residual = h
        h = self.channel_norm(h)
        h = self.channel_mlp(h)
        h = h + residual

        h = self.out_norm(h)
        return self.out_proj(h)
