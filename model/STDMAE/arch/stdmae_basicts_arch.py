import torch
from torch import nn

from ..config.stdmae_config import STDMAEConfig
from .graphwavenet import GraphWaveNet
from .mask import Mask


class STDMAE(nn.Module):
    def __init__(self, config: STDMAEConfig):
        super().__init__()
        if config.input_len is None or config.output_len is None or config.num_nodes is None:
            raise ValueError("STDMAEConfig.input_len/output_len/num_nodes must be set")
        supports = [torch.as_tensor(config.adj, dtype=torch.float32)] if config.adj is not None else None

        mask_args = dict(
            patch_size=int(config.patch_size),
            in_channel=1,
            embed_dim=int(config.mask_embed_dim),
            num_heads=int(config.mask_num_heads),
            mlp_ratio=int(config.mask_mlp_ratio),
            dropout=float(config.mask_dropout),
            mask_ratio=float(config.mask_ratio),
            encoder_depth=int(config.encoder_depth),
            decoder_depth=int(config.decoder_depth),
            mode="forecasting",
        )
        self.tmae = Mask(**mask_args)
        self.smae = Mask(**mask_args)
        self.backend = GraphWaveNet(
            num_nodes=int(config.num_nodes),
            supports=supports,
            dropout=float(config.gw_dropout),
            gcn_bool=True,
            addaptadj=True,
            aptinit=None,
            in_dim=2,
            out_dim=int(config.output_len),
            residual_channels=int(config.gw_residual_channels),
            dilation_channels=int(config.gw_dilation_channels),
            skip_channels=int(config.gw_skip_channels),
            end_channels=int(config.gw_end_channels),
            kernel_size=2,
            blocks=int(config.gw_blocks),
            layers=int(config.gw_layers),
        )
        self.short_term_len = min(int(config.output_len), int(config.input_len))
        self.time_of_day_size = 288

    def forward(self, inputs: torch.Tensor, inputs_timestamps: torch.Tensor = None) -> torch.Tensor:
        if inputs.dim() == 3:
            x = inputs.unsqueeze(-1)
        else:
            x = inputs
        long_history = x[..., [0]]
        short_value = x[:, -self.short_term_len :, :, [0]]
        if inputs_timestamps is not None and inputs_timestamps.dim() == 3:
            ts = inputs_timestamps[:, -self.short_term_len :, :]
            if float(ts.detach().max().item()) <= 1.5:
                short_time = ts[:, :, 0:1].unsqueeze(2).expand(-1, -1, x.shape[2], -1)
            else:
                short_time = ((ts[:, :, 0:1] % self.time_of_day_size) / float(self.time_of_day_size)).unsqueeze(2).expand(-1, -1, x.shape[2], -1)
            short_time = short_time.to(short_value.dtype)
        else:
            b, t, n, _ = short_value.shape
            base = torch.arange(t, device=x.device).view(1, t, 1, 1).expand(b, -1, n, -1)
            short_time = (base % self.time_of_day_size).to(short_value.dtype) / float(self.time_of_day_size)
        short_history = torch.cat([short_value, short_time], dim=-1)  # [B, T, N, 2]
        hs_t = self.tmae(long_history)
        hs_s = self.smae(long_history)
        hs = torch.cat((hs_t, hs_s), dim=-1)
        hs = hs[:, :, -1, :]
        y = self.backend(short_history, hidden_states=hs)
        if y.dim() == 3:
            return y
        if y.dim() == 4 and y.shape[-1] == 1:
            return y.squeeze(-1)
        return y
