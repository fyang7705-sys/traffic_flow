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
                )
                bias = torch.matmul(embeddings, self.bias_pool)
                x_gconv = (
                    torch.einsum("bni,bio->bno", x_g, weights) + bias[:, None, :]
                )
            elif self.meta_axis == "S":
                weights = torch.einsum(
                    "nd,dio->nio", embeddings, self.weights_pool
                )
                bias = torch.matmul(embeddings, self.bias_pool)
                x_gconv = (
                    torch.einsum("bni,nio->bno", x_g, weights) + bias
                )
            elif self.meta_axis == "ST":
                weights = torch.einsum(
                    "bnd,dio->bnio", embeddings, self.weights_pool
                )
                bias = torch.einsum("bnd,do->bno", embeddings, self.bias_pool)
                x_gconv = (
                    torch.einsum("bni,bnio->bno", x_g, weights) + bias
                )
        else:
            x_gconv = torch.einsum("bni,io->bno", x_g, self.weights) + self.bias

        return x_gconv


class HimGCRU(nn.Module):
    """HimNet GCRU: GCN + GRU-style recurrent unit."""

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
            batch_size, self.num_nodes, self.hidden_dim,
            device=device, dtype=dtype,
        )

    def forward(self, x, support, embeddings, init_state=None):
        if x.dim() != 4:
            raise ValueError(
                f"HimGCRU expects x with shape (B, T, N, C), got {tuple(x.shape)}"
            )

        B, T, N, _ = x.shape
        if N != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, but got N={N}")

        if init_state is None:
            state = self.init_hidden_state(B, device=x.device, dtype=x.dtype)
        else:
            if init_state.shape != (B, N, self.hidden_dim):
                raise ValueError(
                    f"init_state shape mismatch: got {tuple(init_state.shape)}"
                )
            state = init_state

        outputs = []
        for t in range(T):
            x_t = x[:, t, :, :]
            input_and_state = torch.cat((x_t, state), dim=-1)
            z_r = torch.sigmoid(self.gate(input_and_state, support, embeddings))
            z, r = torch.split(z_r, self.hidden_dim, dim=-1)
            candidate_input = torch.cat((x_t, z * state), dim=-1)
            hc = torch.tanh(self.update(candidate_input, support, embeddings))
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
        current_input = x
        output_hidden = []
        for cell in self.cells:
            current_input, state = cell(current_input, support, embeddings, init_state=None)
            output_hidden.append(state)
        return current_input, output_hidden



class HimNet(nn.Module):
    """
    改进后的 HimNet。

    本版的关键改动 (相对原版的差异):

    1. iTransformer 时间嵌入输出节点级 (B, N, time_dim)。
    2. encoder_t 改为 meta_axis="ST", 每节点个性化时间响应。
    3. 双流融合用门控 g*h_s + (1-g)*h_t。
    4. 解码端改为并行多步预测头, 去掉自回归 rollout。
    5. node_embedding 共享主干 + 任务专用投影,
       解耦四个使用点之间的梯度冲突 (本次新增):
         - encoder_s   GCN 元权重生成   <-  W_s   @ E_base
         - graph topo  adaptive_support <-  W_g   @ E_base
         - iTrans      token 身份编码   <-  W_id  @ E_base
         - graph fuse  cross-attn query <-  W_f   @ E_base
       这样每个角色都有独立子空间, 共享主干仍编码"我是同一个节点"的归纳偏置。

    新增超参:
        node_base_dim:  共享主干维度。None 时等于 node_embedding_dim
                        (此时仅起解耦作用,不扩容)。
        decouple_node_embedding: 是否启用任务投影 (False 时退化为原版共享行为)。
    """

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

        use_time_embedding=True,
        use_graph_fusion=True,
        static_supports=None,
        # iTransformer time embedding
        time_d_model=64,
        time_nhead=4,
        time_layers=1,
        time_ff_dim=None,
        time_dropout=0.1,
        # 共享主干 + 任务投影
        node_base_dim=None,
        decouple_node_embedding=True,
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
        self.use_time_embedding = use_time_embedding
        self.use_graph_fusion = use_graph_fusion
        self.static_supports = static_supports
        self.decouple_node_embedding = decouple_node_embedding

        self.time_embedding_dim = tod_embedding_dim + dow_embedding_dim

        # ------------------------------------------------------------------
        # 共享主干 + 任务投影
        # ------------------------------------------------------------------
        if node_base_dim is None:
            node_base_dim = node_embedding_dim
        self.node_base_dim = node_base_dim

        # 共享主干: 编码"通用节点表示"
        self.node_embedding = nn.Parameter(
            torch.empty(self.num_nodes, self.node_base_dim)
        )
        nn.init.xavier_normal_(self.node_embedding)

        if self.decouple_node_embedding:
            # 4 个任务专用投影, 各自吸收任务特异梯度
            #   _s:    encoder_s 的 GCN 元权重 (拓扑/功能角色)
            #   _graph: adaptive_support 拓扑构造 (拓扑相似度)
            #   _id:    iTrans token 身份 (区分度/正交性)
            #   _fuse:  GraphFusion query 源 (静态图中的语义)
            self.node_proj_s = nn.Linear(node_base_dim, node_embedding_dim)
            self.node_proj_graph = nn.Linear(node_base_dim, node_embedding_dim)
            self.node_proj_id = nn.Linear(node_base_dim, node_embedding_dim)
            self.node_proj_fuse = nn.Linear(node_base_dim, node_embedding_dim)
        else:
            # 退化为原版: 所有使用点共享 self.node_embedding
            # 此时要求 node_base_dim == node_embedding_dim
            if node_base_dim != node_embedding_dim:
                raise ValueError(
                    "decouple_node_embedding=False 要求 "
                    "node_base_dim == node_embedding_dim, "
                    f"got base={node_base_dim}, emb={node_embedding_dim}"
                )

        # ------------------------------------------------------------------
        # 双流编码器
        # ------------------------------------------------------------------
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
            meta_axis="ST",
        )

        # 双流门控融合
        self.fusion_gate = nn.Linear(2 * hidden_dim, hidden_dim)

        # iTransformer 节点级时间嵌入
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

        # 图融合模块
        self.graph_fusion = CrossAttentionGraphFusion(
            num_nodes=num_nodes,
            node_embedding_dim=node_embedding_dim,
            attn_dim=node_embedding_dim,
            fusion_hidden_dim=16,
            dropout=time_dropout,
        )

        # 并行多步预测头
        self.multi_step_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(time_dropout),
            nn.Linear(hidden_dim, out_steps * output_dim),
        )

    # ----------------------------------------------------------------------
    # 任务专用 embedding 提取器
    # ----------------------------------------------------------------------
    def _task_embeddings(self):
        """返回 4 个任务专用 embedding (N, node_embedding_dim)."""
        if self.decouple_node_embedding:
            return (
                self.node_proj_s(self.node_embedding),
                self.node_proj_graph(self.node_embedding),
                self.node_proj_id(self.node_embedding),
                self.node_proj_fuse(self.node_embedding),
            )
        else:
            E = self.node_embedding
            return E, E, E, E

    def forward(self, x):
        """
        x: B, T_in, N, input_dim
        return: B, out_steps, N, output_dim
        """
        B, T_in, N, _ = x.shape

        # 4 个任务专用 embedding
        E_s, E_graph, E_id, E_fuse = self._task_embeddings()

        # --------------------------------------------------------------
        # 1. iTransformer 节点级时间嵌入 (用 E_id)
        # --------------------------------------------------------------
        if self.use_time_embedding:
            # print("Using iTransformerGlobalTimeEmbedding for node-level time embedding.")
            time_embedding_node = self.time_pattern_embedding(
                x, node_embedding=E_id
            )
        else:
            time_embedding_node = torch.zeros(
                B, N, self.time_embedding_dim,
                device=x.device, dtype=x.dtype,
            )

        # --------------------------------------------------------------
        # 2. 自适应图 (用 E_graph) + 静态图融合 (用 E_fuse)
        # --------------------------------------------------------------
        adaptive_support = torch.softmax(
            torch.relu(E_graph @ E_graph.T), dim=-1
        )

        if self.static_supports is not None and self.use_graph_fusion:
            # print("Using CrossAttentionGraphFusion to fuse adaptive and static supports.")
            support_s = self.graph_fusion(
                node_embedding=E_fuse,
                adaptive_support=adaptive_support,
                static_support=self.static_supports,
            )
        else:
            support_s = adaptive_support

        # --------------------------------------------------------------
        # 3. 双流编码 (encoder_s 用 E_s, encoder_t 用 time_embedding_node)
        # --------------------------------------------------------------
        h_s_seq, _ = self.encoder_s(x, support_s, E_s)
        h_t_seq, _ = self.encoder_t(x, support_s, time_embedding_node)

        # --------------------------------------------------------------
        # 4. 门控融合
        # --------------------------------------------------------------
        gate = torch.sigmoid(
            self.fusion_gate(torch.cat([h_s_seq, h_t_seq], dim=-1))
        )
        h_fuse_seq = gate * h_s_seq + (1.0 - gate) * h_t_seq
        h_last = h_fuse_seq[:, -1, :, :]

        # --------------------------------------------------------------
        # 5. 并行多步预测
        # --------------------------------------------------------------
        out = self.multi_step_head(h_last)
        out = out.view(B, N, self.out_steps, self.output_dim)
        out = out.permute(0, 2, 1, 3).contiguous()
        return out


if __name__ == "__main__":
    try:
        from torchinfo import summary
        model = HimNet(num_nodes=207, input_dim=3).cpu()
        summary(model, [[64, 12, 207, 3]], device="cpu")
    except ImportError:
        model = HimNet(num_nodes=207, input_dim=3).cpu()
        x = torch.randn(2, 12, 207, 3)
        y = model(x)
        print(y.shape)

