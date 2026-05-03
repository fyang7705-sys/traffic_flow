import torch
from torch import nn

from ..config.bigst_config import BigSTConfig
from .model import Model


class BigST(nn.Module):
    def __init__(self, config: BigSTConfig):
        super().__init__()
        if config.input_len is None or config.output_len is None or config.num_nodes is None:
            raise ValueError("BigSTConfig.input_len/output_len/num_nodes must be set")
        supports = None
        if config.adj is not None:
            supports = [torch.as_tensor(config.adj, dtype=torch.float32)]
        self.time_of_day_size = int(config.time_of_day_size)
        self.day_of_week_size = int(config.day_of_week_size)
        self.model = Model(
            seq_num=int(config.input_len),
            in_dim=int(config.in_dim),
            out_dim=int(config.output_len),
            hid_dim=int(config.hid_dim),
            num_nodes=int(config.num_nodes),
            tau=float(config.tau),
            random_feature_dim=int(config.random_feature_dim),
            node_emb_dim=int(config.node_emb_dim),
            time_emb_dim=int(config.time_emb_dim),
            use_residual=bool(config.use_residual),
            use_bn=bool(config.use_bn),
            use_spatial=bool(config.use_spatial),
            use_long=False,
            dropout=float(config.dropout),
            time_of_day_size=self.time_of_day_size,
            day_of_week_size=self.day_of_week_size,
            supports=supports,
            edge_indices=None,
        )

    def forward(self, inputs: torch.Tensor, inputs_timestamps: torch.Tensor = None) -> torch.Tensor:
        if inputs.dim() == 3:
            x = inputs.unsqueeze(-1)
        else:
            x = inputs
        b, t, n, _ = x.shape
        value = x[..., [0]]
        if inputs_timestamps is not None and inputs_timestamps.dim() == 3:
            ts = inputs_timestamps
            if float(ts.detach().max().item()) <= 1.5:
                tod = ts[:, :, 0:1]
                dow = ts[:, :, 1:2] * 7 if ts.shape[-1] > 1 else torch.zeros_like(tod)
            else:
                tod = (ts[:, :, 0:1] % self.time_of_day_size) / float(self.time_of_day_size)
                dow = (ts[:, :, 1:2] % 7) if ts.shape[-1] > 1 else torch.zeros_like(tod)
        else:
            base = torch.arange(t, device=x.device).view(1, t, 1, 1).expand(b, -1, n, -1)
            tod = (base % self.time_of_day_size) / float(self.time_of_day_size)
            dow = ((base // self.time_of_day_size) % 7).float()
        feat = torch.cat([value, tod.to(value.dtype), dow.to(value.dtype)], dim=-1)  # [B,T,N,3]
        out = self.model(feat.permute(0, 2, 1, 3))
        y = out["prediction"]
        if y.dim() == 4 and y.shape[-1] == 1:
            y = y.squeeze(-1)
        return y
