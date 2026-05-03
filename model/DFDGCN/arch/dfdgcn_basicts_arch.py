import torch
from torch import nn

from ..config.dfdgcn_config import DFDGCNConfig
from .dfdgcn_arch import DFDGCN as LegacyDFDGCN


class DFDGCN(nn.Module):
    def __init__(self, config: DFDGCNConfig):
        super().__init__()
        if config.input_len is None or config.output_len is None or config.num_nodes is None:
            raise ValueError("DFDGCNConfig.input_len/output_len/num_nodes must be set")
        supports = None
        if config.adj is not None:
            supports = [torch.as_tensor(config.adj, dtype=torch.float32)]
        self.time_of_day_size = int(config.time_of_day_size)
        self.day_of_week_size = int(config.day_of_week_size)
        self.model = LegacyDFDGCN(
            out_dim=int(config.output_len),
            seq_len=int(config.input_len),
            dropout=float(config.dropout),
            blocks=int(config.blocks),
            layers=int(config.layers),
            gcn_bool=bool(config.gcn_bool),
            addaptadj=bool(config.addaptadj),
            supports=supports,
            fft_emb=int(config.fft_emb),
            subgraph=int(config.subgraph),
            identity_emb=int(config.identity_emb),
            hidden_emb=int(config.hidden_emb),
            num_nodes=int(config.num_nodes),
            time_of_day_size=self.time_of_day_size,
            day_of_week_size=self.day_of_week_size,
            a=float(config.a),
            affine=bool(config.affine),
            # Legacy DFDGCN uses only the first 2 channels for start_conv.
            in_dim=2,
            residual_channels=int(config.residual_channels),
            dilation_channels=int(config.dilation_channels),
            skip_channels=int(config.skip_channels),
            end_channels=int(config.end_channels),
            kernel_size=int(config.kernel_size),
            aptinit=None,
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
                dow = ts[:, :, 1:2] if ts.shape[-1] > 1 else torch.zeros_like(tod)
            else:
                tod = (ts[:, :, 0:1] % self.time_of_day_size) / float(self.time_of_day_size)
                dow = (ts[:, :, 1:2] % 7) / 7.0 if ts.shape[-1] > 1 else torch.zeros_like(tod)
        else:
            base = torch.arange(t, device=x.device).view(1, t, 1, 1).expand(b, -1, n, -1)
            tod = (base % self.time_of_day_size) / float(self.time_of_day_size)
            dow = ((base // self.time_of_day_size) % 7) / 7.0
        feat = torch.cat([value, tod.to(value.dtype), dow.to(value.dtype)], dim=-1)
        y = self.model(feat, feat[:, -1:, ...], 0, 0, self.training)
        if y.dim() == 4 and y.shape[-1] == 1:
            y = y.squeeze(-1)
        elif y.dim() == 4 and y.shape[1] == 1:
            y = y.squeeze(1)
        return y
