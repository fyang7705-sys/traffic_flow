import torch
from torch import nn

class ITransformerGlobalTimeEmbedding(nn.Module):
    """
    iTransformer-style global time embedding.

    输入:
        x: B, T, N, C
        node_embedding: N, node_embedding_dim

    处理:
        1. 节点 N 作为 token
        2. 每个节点 token 的特征是 T * C
        3. 使用共享的 node_embedding 作为节点身份编码
        4. Transformer 在节点 token 间建模全局相关性
        5. 池化得到 B, output_dim
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
        num_layers=1,
        dim_feedforward=None,
        dropout=0.1,
        use_shared_node_embedding=True,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by nhead={nhead}"
            )

        self.in_steps = in_steps
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.use_shared_node_embedding = use_shared_node_embedding

        # 每个节点 token 的输入特征是 T * C
        self.value_embedding = nn.Linear(in_steps * input_dim, d_model)

        if use_shared_node_embedding:
            self.node_embedding_proj = nn.Linear(node_embedding_dim, d_model)
        else:
            self.node_embedding_proj = None

        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(d_model)

        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, x, node_embedding=None):
        """
        x:
            B, T, N, C

        node_embedding:
            N, node_embedding_dim

        return:
            time_embedding:
                B, output_dim
        """
        B, T, N, C = x.shape

        if T != self.in_steps:
            raise ValueError(
                f"Expected T={self.in_steps}, but got T={T}."
            )

        if N != self.num_nodes:
            raise ValueError(
                f"Expected N={self.num_nodes}, but got N={N}."
            )

        if C != self.input_dim:
            raise ValueError(
                f"Expected C={self.input_dim}, but got C={C}."
            )

        # B, T, N, C -> B, N, T, C
        x = x.permute(0, 2, 1, 3).contiguous()

        # B, N, T, C -> B, N, T*C
        node_tokens = x.view(B, N, T * C)

        # B, N, T*C -> B, N, d_model
        node_tokens = self.value_embedding(node_tokens)

        # 复用 HimNet 的 self.node_embedding 作为节点身份编码
        if self.use_shared_node_embedding:
            if node_embedding is None:
                raise ValueError(
                    "node_embedding must be provided when use_shared_node_embedding=True."
                )

            # N, node_embedding_dim -> N, d_model
            node_identity = self.node_embedding_proj(node_embedding)

            # N, d_model -> 1, N, d_model
            node_identity = node_identity.unsqueeze(0)

            node_tokens = node_tokens + node_identity

        # B, N, d_model
        node_tokens = self.encoder(node_tokens)
        node_tokens = self.norm(node_tokens)

        # B, d_model
        global_pattern = node_tokens.mean(dim=1)

        # B, output_dim
        time_embedding = self.out_proj(global_pattern)

        return time_embedding
