import numpy as np
import torch
import torch.nn as nn

from .iTrans import TimeEmbedding
from .GraphFusion import GraphFusion


# ----------------------------------------------------------------------
# HimGCN
# ----------------------------------------------------------------------
class HimGCN(nn.Module):
    def __init__(self, input_dim, output_dim, cheb_k, embed_dim, meta_axis=None):
        super().__init__()
        self.cheb_k = cheb_k
        self.meta_axis = meta_axis.upper() if meta_axis else None

        if meta_axis:
            self.weights_pool = nn.init.xavier_normal_(nn.Parameter(
                torch.FloatTensor(embed_dim, cheb_k * input_dim, output_dim)))
            self.bias_pool = nn.init.xavier_normal_(nn.Parameter(
                torch.FloatTensor(embed_dim, output_dim)))
        else:
            self.weights = nn.init.xavier_normal_(nn.Parameter(
                torch.FloatTensor(cheb_k * input_dim, output_dim)))
            self.bias = nn.init.constant_(nn.Parameter(
                torch.FloatTensor(output_dim)), val=0)

    def forward(self, x, support, embeddings):
        x_g = []
        if support.dim() == 2:
            graph_list = [torch.eye(support.shape[0]).to(support.device), support]
            for k in range(2, self.cheb_k):
                graph_list.append(torch.matmul(2 * support, graph_list[-1]) - graph_list[-2])
            for graph in graph_list:
                x_g.append(torch.einsum("nm,bmc->bnc", graph, x))
        elif support.dim() == 3:
            graph_list = [
                torch.eye(support.shape[1]).repeat(support.shape[0], 1, 1).to(support.device),
                support,
            ]
            for k in range(2, self.cheb_k):
                graph_list.append(torch.matmul(2 * support, graph_list[-1]) - graph_list[-2])
            for graph in graph_list:
                x_g.append(torch.einsum("bnm,bmc->bnc", graph, x))
        x_g = torch.cat(x_g, dim=-1)

        if self.meta_axis:
            if self.meta_axis == "T":
                weights = torch.einsum("bd,dio->bio", embeddings, self.weights_pool)
                bias = torch.matmul(embeddings, self.bias_pool)
                x_gconv = torch.einsum("bni,bio->bno", x_g, weights) + bias[:, None, :]
            elif self.meta_axis == "S":
                weights = torch.einsum("nd,dio->nio", embeddings, self.weights_pool)
                bias = torch.matmul(embeddings, self.bias_pool)
                x_gconv = torch.einsum("bni,nio->bno", x_g, weights) + bias
            elif self.meta_axis == "ST":
                weights = torch.einsum("bnd,dio->bnio", embeddings, self.weights_pool)
                bias = torch.einsum("bnd,do->bno", embeddings, self.bias_pool)
                x_gconv = torch.einsum("bni,bnio->bno", x_g, weights) + bias
        else:
            x_gconv = torch.einsum("bni,io->bno", x_g, self.weights) + self.bias
        return x_gconv


# ----------------------------------------------------------------------
# HimGCRU / HimEncoder
# ----------------------------------------------------------------------
class HimGCRU(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, cheb_k, embed_dim, meta_axis="S"):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = output_dim
        self.gate = HimGCN(input_dim + output_dim, 2 * output_dim, cheb_k, embed_dim, meta_axis)
        self.update = HimGCN(input_dim + output_dim, output_dim, cheb_k, embed_dim, meta_axis)

    def init_hidden_state(self, batch_size, device=None, dtype=None):
        return torch.zeros(batch_size, self.num_nodes, self.hidden_dim,
                           device=device, dtype=dtype)

    def forward(self, x, support, embeddings, init_state=None):
        if x.dim() != 4:
            raise ValueError(f"HimGCRU expects (B,T,N,C), got {tuple(x.shape)}")
        B, T, N, _ = x.shape
        if N != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got N={N}")

        state = init_state if init_state is not None else \
                self.init_hidden_state(B, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(T):
            x_t = x[:, t, :, :]
            input_and_state = torch.cat((x_t, state), dim=-1)
            z_r = torch.sigmoid(self.gate(input_and_state, support, embeddings))
            z, r = torch.split(z_r, self.hidden_dim, dim=-1)
            candidate = torch.cat((x_t, z * state), dim=-1)
            hc = torch.tanh(self.update(candidate, support, embeddings))
            state = r * state + (1.0 - r) * hc
            outputs.append(state)
        return torch.stack(outputs, dim=1), state


class HimEncoder(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, cheb_k, num_layers,
                 embed_dim, meta_axis="S"):
        super().__init__()
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            HimGCRU(
                num_nodes,
                input_dim if i == 0 else output_dim,
                output_dim, cheb_k, embed_dim, meta_axis,
            )
            for i in range(num_layers)
        ])

    def forward(self, x, support, embeddings):
        current_input = x
        output_hidden = []
        for cell in self.cells:
            current_input, state = cell(current_input, support, embeddings, init_state=None)
            output_hidden.append(state)
        return current_input, output_hidden


# ----------------------------------------------------------------------
# HimNet (简化版, 不含 region 特征注入)
# ----------------------------------------------------------------------
class HimNet(nn.Module):
    """
    简化后的 HimNet。

    保留的核心改进:
      - 双流编码器 (encoder_s 用 S 模式, encoder_t 用 ST 模式)
      - 节点级时间嵌入 (SimpleTimeEmbedding, MLP-Mixer 风格)
      - 双流门控融合 (g*h_s + (1-g)*h_t)
      - 并行多步预测头 (替代自回归 rollout)
      - node_embedding 共享主干 + 4 个任务投影 (S / graph / id / fuse)
      - 多源 static prior 软混合 (在 HimNet 内部完成)
      - SimpleGraphFusion (2 个标量软混合)

    使用方式:
        model = HimNet(
            num_nodes=202,
            input_dim=3,
            hidden_dim=64,
            static_supports=adj,                 # 单张拓扑图
            extra_static_supports=[adj_2, adj_3] # (可选) 额外静态先验
        )
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
        # 兼容保留 (本版不使用)
        st_embedding_dim=16,
        tf_decay_steps=4000,
        use_teacher_forcing=True,
        # 模块开关
        use_time_embedding=True,
        use_graph_fusion=True,
        # 静态先验图
        static_supports=None,
        extra_static_supports=None,
        # 时间模式 (SimpleTimeEmbedding)
        time_d_model=32,
        time_dropout=0.1,
        # 兼容保留 (本版不使用)
        time_nhead=4,
        time_layers=1,
        time_ff_dim=None,
        # node embedding 解耦
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
        self.decouple_node_embedding = decouple_node_embedding

        self.time_embedding_dim = tod_embedding_dim + dow_embedding_dim

        # ------------------------------------------------------------------
        # 共享主干 + 任务投影
        # ------------------------------------------------------------------
        if node_base_dim is None:
            node_base_dim = node_embedding_dim
        self.node_base_dim = node_base_dim

        self.node_embedding = nn.Parameter(
            torch.empty(self.num_nodes, self.node_base_dim)
        )
        nn.init.xavier_normal_(self.node_embedding)

        if self.decouple_node_embedding:
            self.node_proj_s = nn.Linear(node_base_dim, node_embedding_dim)
            self.node_proj_graph = nn.Linear(node_base_dim, node_embedding_dim)
            self.node_proj_id = nn.Linear(node_base_dim, node_embedding_dim)
            self.node_proj_fuse = nn.Linear(node_base_dim, node_embedding_dim)
        else:
            if node_base_dim != node_embedding_dim:
                raise ValueError(
                    "decouple_node_embedding=False 要求 node_base_dim == node_embedding_dim"
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
            meta_axis="T",
        )

        # 双流门控融合
        self.fusion_gate = nn.Linear(2 * hidden_dim, hidden_dim)

        # ------------------------------------------------------------------
        # MLP-Mixer 风格时间模式
        # ------------------------------------------------------------------
        self.time_pattern_embedding = TimeEmbedding(
            in_steps=in_steps,
            num_nodes=num_nodes,
            input_dim=input_dim,
            output_dim=self.time_embedding_dim,
            node_embedding_dim=node_embedding_dim,
            d_model=time_d_model,
            dropout=time_dropout,
        )

        # ------------------------------------------------------------------
        # 简化图融合 (2 个标量软混合)
        # ------------------------------------------------------------------
        self.graph_fusion = GraphFusion(
            num_nodes=num_nodes,
            static_bias_init=1.0,
        )

        # ------------------------------------------------------------------
        # 并行多步预测头
        # ------------------------------------------------------------------
        self.multi_step_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(time_dropout),
            nn.Linear(hidden_dim, out_steps * output_dim),
        )

        # ------------------------------------------------------------------
        # 多源 static prior 收集 + 软混合权重
        # ------------------------------------------------------------------
        static_priors = []
        if static_supports is not None:
            if isinstance(static_supports, (list, tuple)):
                for s in static_supports:
                    static_priors.append(torch.as_tensor(s).float())
            else:
                static_priors.append(torch.as_tensor(static_supports).float())
        if extra_static_supports is not None:
            for s in extra_static_supports:
                static_priors.append(torch.as_tensor(s).float())

        if len(static_priors) > 0:
            stacked = torch.stack(static_priors, dim=0)   # K, N, N
            self.register_buffer("static_priors", stacked)
            self.num_static_priors = stacked.shape[0]
            if self.num_static_priors > 1:
                self.static_mixer_logits = nn.Parameter(
                    torch.zeros(self.num_static_priors)
                )
            else:
                self.static_mixer_logits = None
        else:
            self.static_priors = None
            self.num_static_priors = 0
            self.static_mixer_logits = None

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    def _task_embeddings(self):
        E = self.node_embedding
        if self.decouple_node_embedding:
            return (
                self.node_proj_s(E),
                self.node_proj_graph(E),
                self.node_proj_id(E),
                self.node_proj_fuse(E),
            )
        return E, E, E, E

    def _mixed_static_support(self):
        if self.static_priors is None:
            return None
        if self.num_static_priors == 1:
            return self.static_priors[0]
        w = torch.softmax(self.static_mixer_logits, dim=0)
        return (w[:, None, None] * self.static_priors).sum(dim=0)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x, labels=None, batches_seen=None):
        """
        x: B, T_in, N, input_dim
        return: B, out_steps, N, output_dim
        """
        B, T_in, N, _ = x.shape

        E_s, E_graph, E_id, E_fuse = self._task_embeddings()

        # 1. 时间嵌入
        if self.use_time_embedding:
            time_embedding = self.time_pattern_embedding(x, node_embedding=E_id)
        else:
            time_embedding = torch.zeros(
                B, self.time_embedding_dim, device=x.device, dtype=x.dtype,
            )

        # 2. 自适应图 + 静态图融合
        adaptive_support = torch.softmax(
            torch.relu(E_graph @ E_graph.T), dim=-1
        )

        mixed_static = self._mixed_static_support()
        if mixed_static is not None and self.use_graph_fusion:
            support_s = self.graph_fusion(
                node_embedding=E_fuse,
                adaptive_support=adaptive_support,
                static_support=mixed_static,
            )
        else:
            support_s = adaptive_support

        # 3. 双流编码
        h_s_seq, _ = self.encoder_s(x, support_s, E_s)
        h_t_seq, _ = self.encoder_t(x, support_s, time_embedding)

        # 4. 门控融合
        gate = torch.sigmoid(
            self.fusion_gate(torch.cat([h_s_seq, h_t_seq], dim=-1))
        )
        h_fuse_seq = gate * h_s_seq + (1.0 - gate) * h_t_seq
        h_last = h_fuse_seq[:, -1, :, :]

        # 5. 并行多步预测
        out = self.multi_step_head(h_last)
        out = out.view(B, N, self.out_steps, self.output_dim)
        return out.permute(0, 2, 1, 3).contiguous()

    # ------------------------------------------------------------------
    # 诊断: 训练后查看融合权重
    # ------------------------------------------------------------------
    def inspect_priors(self):
        info = {
            "num_static_priors": self.num_static_priors,
            "graph_fusion_weights": self.graph_fusion.get_weights(),
        }
        if self.num_static_priors > 1:
            info["static_prior_weights"] = torch.softmax(
                self.static_mixer_logits, dim=0
            ).detach().cpu().tolist()
        return info


if __name__ == "__main__":
    model = HimNet(num_nodes=202, input_dim=3, hidden_dim=64).cpu()
    x = torch.randn(2, 12, 202, 3)
    y = model(x)
    print(f"output: {y.shape}")
    print(f"total params: {sum(p.numel() for p in model.parameters()):,}")

