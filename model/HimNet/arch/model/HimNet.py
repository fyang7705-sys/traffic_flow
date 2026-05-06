import numpy as np
import torch
import torch.nn as nn
from .iTrans import ITransformerGlobalTimeEmbedding
from .GraphFusion import CrossAttentionGraphFusion


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


class HimGCRU(nn.Module):
    """
    Original HimNet GCRU: GCN + GRU-style recurrent unit.

    This class keeps the sequence-level interface used by the modified code:
        x: B, T, N, input_dim
        init_state: optional B, N, hidden_dim

    Internally, every time step follows the original HimNet implementation:

        input_and_state = concat(x_t, state)
        z, r = sigmoid(HimGCN(input_and_state)).chunk(2)
        hc = tanh(HimGCN(concat(x_t, z * state)))
        state = r * state + (1 - r) * hc

    The Transformer arguments are kept only for compatibility with the
    previous GCN + Transformer version. They are not used here.
    """

    def __init__(
        self,
        num_nodes,
        input_dim,
        output_dim,
        cheb_k,
        embed_dim,
        meta_axis="S",
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = output_dim

        self.gate = HimGCN(
            input_dim=input_dim + output_dim,
            output_dim=2 * output_dim,
            cheb_k=cheb_k,
            embed_dim=embed_dim,
            meta_axis=meta_axis,
        )

        self.update = HimGCN(
            input_dim=input_dim + output_dim,
            output_dim=output_dim,
            cheb_k=cheb_k,
            embed_dim=embed_dim,
            meta_axis=meta_axis,
        )

    def init_hidden_state(self, batch_size, device=None, dtype=None):
        return torch.zeros(
            batch_size,
            self.num_nodes,
            self.hidden_dim,
            device=device,
            dtype=dtype,
        )

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
            outputs:
                B, T, N, hidden_dim

            state:
                B, N, hidden_dim
        """
        if x.dim() != 4:
            raise ValueError(
                f"HimGCRU expects x with shape (B, T, N, C), got {tuple(x.shape)}"
            )

        B, T, N, _ = x.shape

        if N != self.num_nodes:
            raise ValueError(
                f"Expected num_nodes={self.num_nodes}, but got N={N}"
            )

        if init_state is None:
            state = self.init_hidden_state(
                batch_size=B,
                device=x.device,
                dtype=x.dtype,
            )
        else:
            if init_state.shape != (B, N, self.hidden_dim):
                raise ValueError(
                    "init_state must have shape "
                    f"(B, N, hidden_dim)=({B}, {N}, {self.hidden_dim}), "
                    f"got {tuple(init_state.shape)}"
                )
            state = init_state

        outputs = []

        for t in range(T):
            x_t = x[:, t, :, :]  # B, N, input_dim

            input_and_state = torch.cat((x_t, state), dim=-1)

            z_r = torch.sigmoid(
                self.gate(
                    input_and_state,
                    support,
                    embeddings,
                )
            )

            z, r = torch.split(z_r, self.hidden_dim, dim=-1)

            candidate_input = torch.cat((x_t, z * state), dim=-1)

            hc = torch.tanh(
                self.update(
                    candidate_input,
                    support,
                    embeddings,
                )
            )

            state = r * state + (1.0 - r) * hc
            outputs.append(state)

        outputs = torch.stack(outputs, dim=1)

        return outputs, state


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
        input_dim=1,
        output_dim=1,
        out_steps=12,
        in_steps=12,
        hidden_dim=64,
        num_layers=1,
        cheb_k=2,
        tod_embedding_dim=8,
        dow_embedding_dim=8,
        node_embedding_dim=16,
        st_embedding_dim=16,
        tf_decay_steps=4000,
        use_teacher_forcing=True,
        use_time_embedding=True,
        use_graph_fusion=True,
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
        self.node_embedding_dim = node_embedding_dim
        self.st_embedding_dim = st_embedding_dim
        self.tf_decay_steps = tf_decay_steps
        self.use_teacher_forcing = use_teacher_forcing
        self.use_time_embedding = use_time_embedding
        self.use_graph_fusion = use_graph_fusion
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
        )

        self.encoder_t = HimEncoder(
            num_nodes=num_nodes,
            input_dim=input_dim,
            output_dim=hidden_dim,
            cheb_k=cheb_k,
            num_layers=num_layers,
            embed_dim=self.time_embedding_dim,
            meta_axis="T",
        )

        self.decoder = HimDecoder(
            num_nodes=num_nodes,
            input_dim=output_dim,
            output_dim=hidden_dim,
            cheb_k=cheb_k,
            num_layers=num_layers,
            embed_dim=st_embedding_dim,
            meta_axis="ST",
        )

        self.graph_fusion = CrossAttentionGraphFusion(
            num_nodes=num_nodes,
            node_embedding_dim=node_embedding_dim,
            attn_dim=node_embedding_dim,
            fusion_hidden_dim=16,
            dropout=time_dropout,
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

    def forward(self, x, labels=None, batches_seen=None):
        """
        x:
            B, T_in, N, input_dim

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

        if self.static_supports is not None and self.use_graph_fusion:
            support_s = self.graph_fusion(
                node_embedding=self.node_embedding,
                adaptive_support=adaptive_support,
                static_support=self.static_supports,
            )
        else:
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

        for t in range(self.out_steps):
            # Original GCRU decoder only needs the current input and hidden states.
            # Do not stack historical prefixes here; the history is stored in ht_list.
            h_de, ht_list = self.decoder(
                go,
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
        summary(model, [[64, 12, 207, 3]], device="cpu")
    except ImportError:
        model = HimNet(num_nodes=207).cpu()
        x = torch.randn(2, 12, 207, 3)
        y = model(x)
        print(y.shape)
