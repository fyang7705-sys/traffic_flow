import torch
from torch import nn

from ..config.stdn_config import STDNConfig
from .model import STDN as LegacySTDN
from .utils import get_lpls


class STDN(nn.Module):
    """BasicTS-compatible STDN wrapper."""

    def __init__(self, config: STDNConfig):
        super().__init__()
        if config.input_len is None or config.output_len is None or config.num_nodes is None:
            raise ValueError("STDNConfig.input_len/output_len/num_nodes must be set")

        self.input_len = int(config.input_len)
        self.output_len = int(config.output_len)
        self.num_nodes = int(config.num_nodes)
        self.time_of_day_size = int(config.time_of_day_size)

        args = {
            "Data": {
                "dataset_name": "traffic_flow",
                "num_of_vertices": self.num_nodes,
                "time_slice_size": max(1, int(1440 / self.time_of_day_size)),
            },
            "Training": {
                "L": int(config.L),
                "K": int(config.K),
                "d": int(config.d),
                "num_his": self.input_len,
                "num_pred": self.output_len,
                "in_channels": int(config.in_channels),
                "out_channels": int(config.out_channels),
                "T_miss_len": int(config.T_miss_len),
                "node_miss_rate": float(config.node_miss_rate),
                "reference": int(config.reference),
                "order": int(config.order),
            },
        }
        self.model = LegacySTDN(args=args, bn_decay=float(config.bn_decay))

        if config.lpls is not None:
            lpls = torch.as_tensor(config.lpls, dtype=torch.float32)
        elif config.adj is not None:
            lpls = get_lpls(torch.as_tensor(config.adj, dtype=torch.float32).cpu().numpy())
        else:
            lpls = torch.zeros(self.num_nodes, 32, dtype=torch.float32)
        self.register_buffer("lpls", lpls)

    def _build_te(self, bsz: int, device: torch.device, inputs_timestamps: torch.Tensor = None) -> torch.Tensor:
        slots = self.time_of_day_size
        if inputs_timestamps is None:
            base = torch.arange(self.input_len + self.output_len, device=device).unsqueeze(0).expand(bsz, -1)
            tod = base % slots
            dow = (base // slots) % 7
            return torch.stack([dow, tod], dim=-1)

        if inputs_timestamps.dim() != 3:
            raise ValueError(f"inputs_timestamps must be [B,T,C], got {tuple(inputs_timestamps.shape)}")
        if inputs_timestamps.shape[1] != self.input_len:
            raise ValueError(f"Expected timestamp length={self.input_len}, got {inputs_timestamps.shape[1]}")

        ts = inputs_timestamps
        maxv = float(ts.detach().max().item()) if ts.numel() > 0 else 0.0
        if maxv <= 1.5:
            tod_h = (ts[:, :, 0] * slots).long() % slots
            if ts.shape[-1] > 1:
                dow_h = (ts[:, :, 1] * 7).long() % 7
            else:
                dow_h = torch.zeros_like(tod_h)
        else:
            tod_h = ts[:, :, 0].long() % slots
            if ts.shape[-1] > 1:
                dow_h = ts[:, :, 1].long() % 7
            else:
                dow_h = torch.zeros_like(tod_h)

        last_abs = dow_h[:, -1] * slots + tod_h[:, -1]
        future_abs = last_abs.unsqueeze(1) + torch.arange(1, self.output_len + 1, device=device).unsqueeze(0)
        tod_f = future_abs % slots
        dow_f = (future_abs // slots) % 7

        tod = torch.cat([tod_h, tod_f], dim=1)
        dow = torch.cat([dow_h, dow_f], dim=1)
        return torch.stack([dow, tod], dim=-1)

    def forward(self, inputs: torch.Tensor, inputs_timestamps: torch.Tensor = None) -> torch.Tensor:
        if inputs.dim() == 3:
            x = inputs.unsqueeze(-1)
        elif inputs.dim() == 4:
            x = inputs
        else:
            raise ValueError(f"inputs must be [B,T,N] or [B,T,N,C], got {tuple(inputs.shape)}")
        if x.shape[1] != self.input_len:
            raise ValueError(f"Expected input_len={self.input_len}, got T={x.shape[1]}")
        if x.shape[2] != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got N={x.shape[2]}")

        bsz = x.shape[0]
        te = self._build_te(bsz, x.device, inputs_timestamps)
        y = self.model(x[..., [0]], te, self.lpls.to(x.device), mode="train")
        if y.dim() == 4 and y.shape[-1] == 1:
            y = y.squeeze(-1)
        return y

