import torch
from torch import nn
from typing import Optional

from ..config.himnet_config import HimNetConfig
from .model.HimNet import HimNet as LegacyHimNet


class HimNet(nn.Module):
    def __init__(self, config: HimNetConfig):
        super().__init__()
        if config.input_len is None or config.output_len is None or config.num_nodes is None:
            raise ValueError("HimNetConfig.input_len/output_len/num_nodes must be set")
        self.time_of_day_size = int(config.time_of_day_size)

        static_supports = None
        if getattr(config, "adj", None) is not None:
            static_supports = torch.as_tensor(config.adj, dtype=torch.float32)

        if getattr(config, "in_steps", None) is None:
            in_steps = int(config.input_len)
        else:
            cfg_in_steps = config.in_steps
            assert cfg_in_steps is not None
            in_steps = int(cfg_in_steps)

        self.model = LegacyHimNet(
            num_nodes=int(config.num_nodes),
            input_dim=int(config.input_dim),
            output_dim=int(config.output_dim),
            out_steps=int(config.output_len),
            in_steps=in_steps,
            hidden_dim=int(config.hidden_dim),
            num_layers=int(config.num_layers),
            cheb_k=int(config.cheb_k),
            tod_embedding_dim=int(config.tod_embedding_dim),
            dow_embedding_dim=int(config.dow_embedding_dim),
            node_embedding_dim=int(config.node_embedding_dim),
            st_embedding_dim=int(config.st_embedding_dim),
            tf_decay_steps=int(config.tf_decay_steps),
            use_teacher_forcing=bool(config.use_teacher_forcing),
            use_time_embedding=bool(config.use_time_embedding),
            use_graph_fusion=bool(config.use_graph_fusion),
            static_supports=static_supports,
            time_d_model=int(config.time_d_model),
            time_nhead=int(config.time_nhead),
            time_layers=int(config.time_layers),
            time_ff_dim=(None if config.time_ff_dim is None else int(config.time_ff_dim)),
            time_dropout=float(config.time_dropout),
        )

    def forward(self, inputs: torch.Tensor, inputs_timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
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
                dow = ts[:, :, 1:2] if ts.shape[-1] > 1 else torch.zeros_like(tod)
            else:
                tod = (ts[:, :, 0:1] % self.time_of_day_size) / float(self.time_of_day_size)
                dow = (ts[:, :, 1:2] % 7).float() if ts.shape[-1] > 1 else torch.zeros_like(tod)
        else:
            base = torch.arange(t, device=x.device).view(1, t, 1, 1).expand(b, -1, n, -1)
            tod = (base % self.time_of_day_size) / float(self.time_of_day_size)
            dow = ((base // self.time_of_day_size) % 7).float()
        x_in = torch.cat([value], dim=-1)

        y = self.model(x_in, labels=None, batches_seen=0)
        if y.dim() == 4 and y.shape[-1] == 1:
            y = y.squeeze(-1)
        return y
