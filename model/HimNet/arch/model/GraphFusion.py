import torch
from torch import nn


class GraphFusion(nn.Module):
    """
    最简图融合: 多个候选图的可学习软混合。

    fused = softmax(theta) @ [adaptive, static]
    """

    def __init__(
        self,
        num_nodes,
        node_embedding_dim=None,    # 保留兼容, 不使用
        attn_dim=None,              # 保留兼容, 不使用
        fusion_hidden_dim=None,     # 保留兼容, 不使用
        dropout=None,               # 保留兼容, 不使用
        static_bias_init: float = 1.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes

        # 2 个通道: [adaptive, static]
        # 初始 logit: adaptive=0, static=static_bias_init (正值, 训练初期 static 占优)
        prior = torch.zeros(2)
        prior[1] = float(static_bias_init)
        self.fusion_logits = nn.Parameter(prior)

    def _row_normalize(self, adj, eps=1e-8):
        adj = torch.relu(adj)
        return adj / (adj.sum(dim=-1, keepdim=True) + eps)

    def forward(self, node_embedding, adaptive_support, static_support):
        """
        node_embedding: N, E  (保留接口兼容, 简化版不使用)
        adaptive_support: N, N
        static_support: N, N
        return: N, N
        """
        static_support = static_support.to(
            device=adaptive_support.device,
            dtype=adaptive_support.dtype,
        )
        static_support = self._row_normalize(static_support)

        # softmax 得到混合权重
        w = torch.softmax(self.fusion_logits, dim=0)   # 2,

        fused = w[0] * adaptive_support + w[1] * static_support
        fused = self._row_normalize(fused)
        return fused

    def get_weights(self):
        """诊断接口: 返回当前的混合权重"""
        return torch.softmax(self.fusion_logits, dim=0).detach().cpu().tolist()


class MultiSourceGraphFusion(nn.Module):
    """
    支持任意多个 static prior 的简化融合。

    输入: adaptive_support + 一组 static priors (在 HimNet 中混合后送入)
    """

    def __init__(
        self,
        num_nodes,
        num_static_priors: int = 1,
        static_bias_init: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_channels = 1 + num_static_priors

        prior = torch.zeros(self.num_channels)
        # 给所有 static 通道一个小的正先验
        prior[1:] = static_bias_init / max(num_static_priors, 1)
        self.fusion_logits = nn.Parameter(prior)

    def _row_normalize(self, adj, eps=1e-8):
        adj = torch.relu(adj)
        return adj / (adj.sum(dim=-1, keepdim=True) + eps)

    def forward(self, adaptive_support, static_supports):
        """
        adaptive_support: N, N
        static_supports: list of N, N (长度 = num_static_priors)
        return: N, N
        """
        all_supports = [adaptive_support] + [
            self._row_normalize(s.to(adaptive_support.device, adaptive_support.dtype))
            for s in static_supports
        ]
        # K, N, N
        stacked = torch.stack(all_supports, dim=0)

        w = torch.softmax(self.fusion_logits, dim=0)   # K,
        fused = (w[:, None, None] * stacked).sum(dim=0)
        return self._row_normalize(fused)

    def get_weights(self):
        return torch.softmax(self.fusion_logits, dim=0).detach().cpu().tolist()
