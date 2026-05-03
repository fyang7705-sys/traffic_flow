import math

import torch
from torch import nn

try:
    from positional_encodings.torch_encodings import PositionalEncoding2D  # type: ignore
except Exception:
    PositionalEncoding2D = None


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _sinusoidal_2d_like(x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, P, D]
        _, n, p, d = x.shape
        device = x.device
        dtype = x.dtype

        half = max(1, d // 2)
        div = torch.exp(torch.arange(0, half, 2, device=device, dtype=dtype) * (-math.log(10000.0) / max(1, half)))

        pe_n = torch.zeros(n, half, device=device, dtype=dtype)
        pos_n = torch.arange(0, n, device=device, dtype=dtype).unsqueeze(1)
        pe_n[:, 0::2] = torch.sin(pos_n * div)
        if pe_n.shape[1] > 1:
            pe_n[:, 1::2] = torch.cos(pos_n * div[: pe_n[:, 1::2].shape[1]])

        d2 = d - half
        pe_p = torch.zeros(p, d2, device=device, dtype=dtype)
        if d2 > 0:
            div2 = torch.exp(torch.arange(0, d2, 2, device=device, dtype=dtype) * (-math.log(10000.0) / max(1, d2)))
            pos_p = torch.arange(0, p, device=device, dtype=dtype).unsqueeze(1)
            pe_p[:, 0::2] = torch.sin(pos_p * div2)
            if pe_p.shape[1] > 1:
                pe_p[:, 1::2] = torch.cos(pos_p * div2[: pe_p[:, 1::2].shape[1]])

        pe = torch.zeros(1, n, p, d, device=device, dtype=dtype)
        pe[:, :, :, :half] = pe_n.unsqueeze(1).expand(n, p, half).unsqueeze(0)
        if d2 > 0:
            pe[:, :, :, half:] = pe_p.unsqueeze(0).unsqueeze(0).expand(1, n, p, d2)
        return pe

    def forward(self, input_data, index=None, abs_idx=None):
        """Positional encoding

        Args:
            input_data (torch.tensor): input sequence with shape [B, N, P, d].
            index (list or None): add positional embedding by index.

        Returns:
            torch.tensor: output sequence
        """

        _, _, _, num_feat = input_data.shape
        if PositionalEncoding2D is not None:
            tp_enc_2d = PositionalEncoding2D(num_feat)
            pos = tp_enc_2d(input_data)
        else:
            pos = self._sinusoidal_2d_like(input_data)
        input_data = input_data + pos
        return input_data, pos
