import torch
from torch import nn


class GraphFusion(nn.Module):
    """门控图融合：自适应图与静态图按节点门控融合。

    旧版是全局两个标量权重（softmax(theta)）。
    新版用 node_embedding 生成每个节点的 gate：
        gate_i = sigmoid(MLP(e_i)) in (0,1)
        fused[i, :] = gate_i * adaptive[i, :] + (1-gate_i) * static[i, :]

    这样不同节点可以学习不同的融合偏好。
    """

    def __init__(
        self,
        num_nodes,
        node_embedding_dim=None,
        fusion_hidden_dim: int = 32,
        dropout: float = 0.0,
        static_bias_init: float = 1.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes

        if node_embedding_dim is None:
            # 兼容旧调用：如果没传维度，就退化为“全局门控”
            self.gate_mlp = None
        else:
            self.gate_mlp = nn.Sequential(
                nn.Linear(int(node_embedding_dim), int(fusion_hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(fusion_hidden_dim), 1),
            )

        # 当 gate_mlp 为 None 或 node_embedding 不可用时，使用一个全局可学习 gate
        # 初始化让 static 更占优：gate 越小 static 权重越大
        init_gate = 1.0 / (1.0 + torch.exp(torch.tensor(float(static_bias_init))))
        # 反推 logit，使得 sigmoid(logit)=init_gate
        init_logit = torch.log(init_gate / (1.0 - init_gate))
        self.global_gate_logit = nn.Parameter(init_logit.clone().detach())

    def _row_normalize(self, adj, eps=1e-8):
        adj = torch.relu(adj)
        return adj / (adj.sum(dim=-1, keepdim=True) + eps)

    def forward(self, node_embedding, adaptive_support, static_support):
        """
        node_embedding: (N, E)
        adaptive_support: (N, N)
        static_support: (N, N)
        return: (N, N)
        """
        static_support = static_support.to(
            device=adaptive_support.device,
            dtype=adaptive_support.dtype,
        )
        static_support = self._row_normalize(static_support)

        adaptive_support = self._row_normalize(adaptive_support)

        if self.gate_mlp is not None and node_embedding is not None:
            gate = torch.sigmoid(self.gate_mlp(node_embedding)).to(
                device=adaptive_support.device,
                dtype=adaptive_support.dtype,
            )  # (N, 1)
        else:
            gate = torch.sigmoid(self.global_gate_logit).to(
                device=adaptive_support.device,
                dtype=adaptive_support.dtype,
            )
            gate = gate.view(1, 1).expand(adaptive_support.shape[0], 1)  # (N, 1)

        fused = gate * adaptive_support + (1.0 - gate) * static_support
        fused = self._row_normalize(fused)
        return fused

    def get_weights(self):
        """诊断接口：返回全局 gate（以及如可用则返回每节点 gate 的占位说明）。"""
        return {
            "global_gate": float(torch.sigmoid(self.global_gate_logit).detach().cpu().item()),
        }

