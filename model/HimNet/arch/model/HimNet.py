import math
import numpy as np
import torch
import torch.nn as nn
from .iTrans import ITransformerGlobalTimeEmbedding
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
        fusion_hidden_dim=16,
        dropout=0.1,
    ):
        super().__init__()

        self.num_nodes = num_nodes

        if attn_dim is None:
            attn_dim = node_embedding_dim

        self.q_proj = nn.Linear(node_embedding_dim, attn_dim)
        self.k_proj = nn.Linear(node_embedding_dim, attn_dim)
        self.dropout = nn.Dropout(dropout)

        # 对 adaptive / static / cross 三种图做边级别动态融合
        self.edge_fusion = nn.Sequential(
            nn.Linear(3, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, 3),
        )

    def _row_normalize(self, adj, eps=1e-8):
        adj = torch.relu(adj)
        return adj / (adj.sum(dim=-1, keepdim=True) + eps)

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

        # 静态图先做行归一化，避免尺度和 adaptive_support 不一致
        static_support = self._row_normalize(static_support)

        # -------------------------------------------------------
        # 1. 用静态图聚合 node_embedding，得到静态结构感知的节点表示
        # -------------------------------------------------------
        # N, E
        static_context = torch.matmul(static_support, node_embedding)

        # -------------------------------------------------------
        # 2. 交叉注意力：
        #    query 来自 node_embedding
        #    key 来自 static_context
        # -------------------------------------------------------
        Q = self.q_proj(node_embedding)        # N, D
        K = self.k_proj(static_context)        # N, D

        attn_logits = torch.matmul(Q, K.T) / math.sqrt(Q.shape[-1])

        # N, N
        cross_support = torch.softmax(attn_logits, dim=-1)
        cross_support = self.dropout(cross_support)

        # -------------------------------------------------------
        # 3. 边级别动态融合 adaptive / static / cross
        # -------------------------------------------------------
        candidates = torch.stack(
            [
                adaptive_support,
                static_support,
                cross_support,
            ],
            dim=-1,
        )
        # N, N, 3

        fusion_logits = self.edge_fusion(candidates)
        fusion_weight = torch.softmax(fusion_logits, dim=-1)
        # N, N, 3

        fused_support = (
            fusion_weight[..., 0] * adaptive_support
            + fusion_weight[..., 1] * static_support
            + fusion_weight[..., 2] * cross_support
        )

        fused_support = self._row_normalize(fused_support)

        return fused_support


class HimGCN(nn.Module):
    def __init__(self, input_dim, output_dim, cheb_k, embed_dim, meta_axis=None):
        super().__init__()
        self.cheb_k = cheb_k
        self.meta_axis = meta_axis.upper() if meta_axis else None

        if meta_axis:
            self.weights_pool = nn.init.xavier_normal_(
                nn.Parameter(
                    torch.FloatTensor(embed_dim, cheb_k * input_dim, output_dim)
                )
            )
            self.bias_pool = nn.init.xavier_normal_(
                nn.Parameter(torch.FloatTensor(embed_dim, output_dim))
            )
        else:
            self.weights = nn.init.xavier_normal_(
                nn.Parameter(torch.FloatTensor(cheb_k * input_dim, output_dim))
            )
            self.bias = nn.init.constant_(
                nn.Parameter(torch.FloatTensor(output_dim)), val=0
            )

    def forward(self, x, support, embeddings):
        x_g = []

        if support.dim() == 2:
            graph_list = [torch.eye(support.shape[0]).to(support.device), support]
            for k in range(2, self.cheb_k):
                graph_list.append(
                    torch.matmul(2 * support, graph_list[-1]) - graph_list[-2]
                )
            for graph in graph_list:
                x_g.append(torch.einsum("nm,bmc->bnc", graph, x))
        elif support.dim() == 3:
            graph_list = [
                torch.eye(support.shape[1])
                .repeat(support.shape[0], 1, 1)
                .to(support.device),
                support,
            ]
            for k in range(2, self.cheb_k):
                graph_list.append(
                    torch.matmul(2 * support, graph_list[-1]) - graph_list[-2]
                )
            for graph in graph_list:
                x_g.append(torch.einsum("bnm,bmc->bnc", graph, x))
        x_g = torch.cat(x_g, dim=-1)

        if self.meta_axis:
            if self.meta_axis == "T":
                weights = torch.einsum(
                    "bd,dio->bio", embeddings, self.weights_pool
                )  # B, cheb_k*in_dim, out_dim
                bias = torch.matmul(embeddings, self.bias_pool)  # B, out_dim
                x_gconv = (
                    torch.einsum("bni,bio->bno", x_g, weights) + bias[:, None, :]
                )  # B, N, out_dim
            elif self.meta_axis == "S":
                weights = torch.einsum(
                    "nd,dio->nio", embeddings, self.weights_pool
                )  # N, cheb_k*in_dim, out_dim
                bias = torch.matmul(embeddings, self.bias_pool)
                x_gconv = (
                    torch.einsum("bni,nio->bno", x_g, weights) + bias
                )  # B, N, out_dim
            elif self.meta_axis == "ST":
                weights = torch.einsum(
                    "bnd,dio->bnio", embeddings, self.weights_pool
                )  # B, N, cheb_k*in_dim, out_dim
                bias = torch.einsum("bnd,do->bno", embeddings, self.bias_pool)
                x_gconv = (
                    torch.einsum("bni,bnio->bno", x_g, weights) + bias
                )  # B, N, out_dim

        else:
            x_gconv = torch.einsum("bni,io->bno", x_g, self.weights) + self.bias

        return x_gconv


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        # 1, max_len, d_model
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """
        x: B*N, T, hidden_dim
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len {self.pe.size(1)}."
            )

        return x + self.pe[:, :seq_len, :].to(dtype=x.dtype)


class HimGCRU(nn.Module):
    """
    Transformer replacement for the original HimGCRU.

    Original HimGCRU:
        x_t + state_{t-1} -> gate/update -> state_t

    New version:
        x_{0:T} -> HimGCN for each time step -> Transformer over time -> h_{0:T}

    The input is now a full sequence:
        x: B, T, N, input_dim

    For decoder usage, init_state can be passed as a context token:
        init_state: B, N, hidden_dim
    """

    def __init__(
        self,
        num_nodes,
        input_dim,
        output_dim,
        cheb_k,
        embed_dim,
        meta_axis="S",
        nhead=4,
        num_transformer_layers=1,
        dim_feedforward=None,
        dropout=0.1,
        max_len=1000,
    ):
        super().__init__()

        if output_dim % nhead != 0:
            raise ValueError(
                f"output_dim={output_dim} must be divisible by nhead={nhead}"
            )

        self.num_nodes = num_nodes
        self.hidden_dim = output_dim

        self.gcn = HimGCN(
            input_dim=input_dim,
            output_dim=output_dim,
            cheb_k=cheb_k,
            embed_dim=embed_dim,
            meta_axis=meta_axis,
        )

        self.pos_encoder = TemporalPositionalEncoding(
            d_model=output_dim,
            max_len=max_len,
        )

        if dim_feedforward is None:
            dim_feedforward = output_dim * 4

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )

        self.norm = nn.LayerNorm(output_dim)


    def forward(self, x, support, embeddings, init_state=None):
        """
        x:
            B, T, N, input_dim

        support:
            N, N
            or B, N, N

        embeddings:
            meta_axis == "S":  N, embed_dim
            meta_axis == "T":  B, embed_dim
            meta_axis == "ST": B, N, embed_dim

        init_state:
            None
            or B, N, hidden_dim

        return:
            h:
                B, T, N, hidden_dim

            last_state:
                B, N, hidden_dim
        """
        if x.dim() != 4:
            raise ValueError(
                f"HimGCRU now expects x with shape (B, T, N, C), got {tuple(x.shape)}"
            )

        B, T, N, _ = x.shape

        spatial_outputs = []
        for t in range(T):
            # B, N, hidden_dim
            h_t = self.gcn(
                x[:, t, :, :],
                support,
                embeddings,
            )
            spatial_outputs.append(h_t)

        # B, T, N, hidden_dim
        h = torch.stack(spatial_outputs, dim=1)

        # B, T, N, H -> B, N, T, H -> B*N, T, H
        h = h.permute(0, 2, 1, 3).contiguous()
        h = h.reshape(B * N, T, self.hidden_dim)

        has_context_token = init_state is not None
        if has_context_token:
            if init_state.shape != (B, N, self.hidden_dim):
                raise ValueError(
                    "init_state must have shape "
                    f"(B, N, hidden_dim)=({B}, {N}, {self.hidden_dim}), "
                    f"got {tuple(init_state.shape)}"
                )

            # B, N, H -> B*N, 1, H
            context = init_state.reshape(B * N, 1, self.hidden_dim)
            h = torch.cat([context, h], dim=1)

        h = self.pos_encoder(h)


        # B*N, T(+1), H
        h = self.temporal_encoder(
            h,
            mask=None,
        )

        h = self.norm(h)

        if has_context_token:
            h = h[:, 1:, :]

        # B*N, T, H -> B, N, T, H -> B, T, N, H
        h = h.reshape(B, N, T, self.hidden_dim)
        h = h.permute(0, 2, 1, 3).contiguous()

        last_state = h[:, -1, :, :]

        return h, last_state


class HimEncoder(nn.Module):
    def __init__(
        self,
        num_nodes,
        input_dim,
        output_dim,
        cheb_k,
        num_layers,
        embed_dim,
        meta_axis="S",
        nhead=4,
        num_transformer_layers=1,
        dim_feedforward=None,
        dropout=0.1,
        max_len=1000,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = output_dim

        self.cells = nn.ModuleList(
            [
                HimGCRU(
                    num_nodes=num_nodes,
                    input_dim=input_dim if i == 0 else output_dim,
                    output_dim=output_dim,
                    cheb_k=cheb_k,
                    embed_dim=embed_dim,
                    meta_axis=meta_axis,
                    nhead=nhead,
                    num_transformer_layers=num_transformer_layers,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    max_len=max_len,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x, support, embeddings):
        """
        x:
            B, T, N, C

        return:
            current_input:
                B, T, N, hidden_dim

            output_hidden:
                list[num_layers], each item is B, N, hidden_dim
        """
        current_input = x
        output_hidden = []

        for cell in self.cells:
            current_input, state = cell(
                current_input,
                support,
                embeddings,
                init_state=None,
            )
            output_hidden.append(state)

        return current_input, output_hidden


class HimDecoder(nn.Module):
    def __init__(
        self,
        num_nodes,
        input_dim,
        output_dim,
        cheb_k,
        num_layers,
        embed_dim,
        meta_axis="ST",
        nhead=4,
        num_transformer_layers=1,
        dim_feedforward=None,
        dropout=0.1,
        max_len=1000,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = output_dim

        self.cells = nn.ModuleList(
            [
                HimGCRU(
                    num_nodes=num_nodes,
                    input_dim=input_dim if i == 0 else output_dim,
                    output_dim=output_dim,
                    cheb_k=cheb_k,
                    embed_dim=embed_dim,
                    meta_axis=meta_axis,
                    nhead=nhead,
                    num_transformer_layers=num_transformer_layers,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    max_len=max_len,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x_seq, init_state, support, embeddings):
        """
        x_seq:
            B, T_dec, N, D
            or B, N, D. If B, N, D is passed, a time dimension is added.

        init_state:
            list[num_layers], each item B, N, hidden_dim
            or tensor with shape num_layers, B, N, hidden_dim

        support:
            B, N, N

        embeddings:
            B, N, st_embedding_dim

        return:
            current_output:
                B, N, hidden_dim

            output_hidden:
                list[num_layers], each item B, N, hidden_dim
        """
        if x_seq.dim() == 3:
            x_seq = x_seq.unsqueeze(1)

        if isinstance(init_state, torch.Tensor):
            layer_states = [init_state[i] for i in range(init_state.shape[0])]
        else:
            layer_states = init_state

        current_input = x_seq
        output_hidden = []

        for i in range(self.num_layers):
            current_input, state = self.cells[i](
                current_input,
                support,
                embeddings,
                init_state=layer_states[i],
            )
            output_hidden.append(state)

        # B, T_dec, N, hidden_dim -> B, N, hidden_dim
        current_output = current_input[:, -1, :, :]

        return current_output, output_hidden


class HimNet(nn.Module):
    def __init__(
        self,
        num_nodes,
        input_dim=3,
        output_dim=1,
        out_steps=12,
        in_steps=12,
        hidden_dim=64,
        num_layers=1,
        cheb_k=2,
        ycov_dim=2,
        tod_embedding_dim=8,
        dow_embedding_dim=8,
        node_embedding_dim=16,
        st_embedding_dim=16,
        tf_decay_steps=4000,
        use_teacher_forcing=True,
        use_time_embedding=True,
        transformer_nhead=4,
        transformer_layers=1,
        transformer_ff_dim=None,
        transformer_dropout=0.1,
        static_supports=None,

        # iTransformer time embedding
        time_d_model=64,
        time_nhead=4,
        time_layers=1,
        time_ff_dim=None,
        time_dropout=0.1,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.out_steps = out_steps
        self.in_steps = in_steps
        self.num_layers = num_layers
        self.cheb_k = cheb_k
        self.ycov_dim = ycov_dim
        self.node_embedding_dim = node_embedding_dim
        self.st_embedding_dim = st_embedding_dim
        self.tf_decay_steps = tf_decay_steps
        self.use_teacher_forcing = use_teacher_forcing
        self.use_time_embedding = use_time_embedding
        self.static_supports = static_supports

        self.time_embedding_dim = tod_embedding_dim + dow_embedding_dim

        self.encoder_s = HimEncoder(
            num_nodes=num_nodes,
            input_dim=input_dim,
            output_dim=hidden_dim,
            cheb_k=cheb_k,
            num_layers=num_layers,
            embed_dim=node_embedding_dim,
            meta_axis="S",
            nhead=transformer_nhead,
            num_transformer_layers=transformer_layers,
            dim_feedforward=transformer_ff_dim,
            dropout=transformer_dropout,
            max_len=1000,
        )

        self.encoder_t = HimEncoder(
            num_nodes=num_nodes,
            input_dim=input_dim,
            output_dim=hidden_dim,
            cheb_k=cheb_k,
            num_layers=num_layers,
            embed_dim=self.time_embedding_dim,
            meta_axis="T",
            nhead=transformer_nhead,
            num_transformer_layers=transformer_layers,
            dim_feedforward=transformer_ff_dim,
            dropout=transformer_dropout,
            max_len=1000,
        )

        self.decoder = HimDecoder(
            num_nodes=num_nodes,
            input_dim=output_dim + ycov_dim,
            output_dim=hidden_dim,
            cheb_k=cheb_k,
            num_layers=num_layers,
            embed_dim=st_embedding_dim,
            meta_axis="ST",
            nhead=transformer_nhead,
            num_transformer_layers=transformer_layers,
            dim_feedforward=transformer_ff_dim,
            dropout=transformer_dropout,
            max_len=out_steps + 1,
        )

        self.out_proj = nn.Linear(hidden_dim, output_dim)

        self.time_pattern_embedding = ITransformerGlobalTimeEmbedding(
            in_steps=in_steps,
            num_nodes=num_nodes,
            input_dim=input_dim,
            output_dim=self.time_embedding_dim,
            node_embedding_dim=node_embedding_dim,
            d_model=time_d_model,
            nhead=time_nhead,
            num_layers=time_layers,
            dim_feedforward=time_ff_dim,
            dropout=time_dropout,
            use_shared_node_embedding=True,
        )

        self.node_embedding = nn.Parameter(
            torch.empty(self.num_nodes, self.node_embedding_dim)
        )
        nn.init.xavier_normal_(self.node_embedding)

        self.st_proj = nn.Linear(self.hidden_dim, self.st_embedding_dim)

    def compute_sampling_threshold(self, batches_seen):
        return self.tf_decay_steps / (
            self.tf_decay_steps + np.exp(batches_seen / self.tf_decay_steps)
        )

    def forward(self, x, y_cov, labels=None, batches_seen=None):
        """
        x:
            B, T_in, N, input_dim

        y_cov:
            B, out_steps, N, ycov_dim

        labels:
            optional, B, out_steps, N, output_dim
        """
        batch_size = x.shape[0]

        if self.use_time_embedding:
            time_embedding = self.time_pattern_embedding(x, node_embedding=self.node_embedding)
        else:
            time_embedding = torch.zeros(
                batch_size,
                self.time_embedding_dim,
                device=x.device,
                dtype=x.dtype,
            )

        adaptive_support = torch.softmax(
            torch.relu(self.node_embedding @ self.node_embedding.T),
            dim=-1,
        )

        support_s = adaptive_support

        h_s, _ = self.encoder_s(
            x,
            support_s,
            self.node_embedding,
        )

        h_t, _ = self.encoder_t(
            x,
            support_s,
            time_embedding,
        )

        h_last = (h_s + h_t)[:, -1, :, :]

        st_embedding = self.st_proj(h_last)

        support_st = torch.softmax(
            torch.relu(
                torch.einsum(
                    "bnc,bmc->bnm",
                    st_embedding,
                    st_embedding,
                )
            ),
            dim=-1,
        )

        ht_list = [h_last for _ in range(self.num_layers)]

        go = torch.zeros(
            batch_size,
            self.num_nodes,
            self.output_dim,
            device=x.device,
            dtype=x.dtype,
        )

        out = []
        decoder_inputs = []

        for t in range(self.out_steps):
            decoder_input_t = torch.cat(
                [go, y_cov[:, t, ...]],
                dim=-1,
            )

            decoder_inputs.append(decoder_input_t)

            decoder_input_seq = torch.stack(
                decoder_inputs,
                dim=1,
            )

            h_de, ht_list = self.decoder(
                decoder_input_seq,
                ht_list,
                support_st,
                st_embedding,
            )

            go = self.out_proj(h_de)
            out.append(go)

            if (
                self.training
                and self.use_teacher_forcing
                and labels is not None
                and batches_seen is not None
            ):
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batches_seen):
                    go = labels[:, t, ...]

        output = torch.stack(out, dim=1)

        return output


if __name__ == "__main__":
    try:
        from torchinfo import summary

        model = HimNet(num_nodes=207).cpu()
        summary(model, [[64, 12, 207, 3], [64, 12, 207, 2]], device="cpu")
    except ImportError:
        model = HimNet(num_nodes=207).cpu()
        x = torch.randn(2, 12, 207, 3)
        y_cov = torch.randn(2, 12, 207, 2)
        y = model(x, y_cov)
        print(y.shape)
