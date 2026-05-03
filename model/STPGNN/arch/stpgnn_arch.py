import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.stpgnn_config import STPGNNConfig


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,nwv->ncwl', (x, A))
        return x.contiguous()

class pconv(nn.Module):
    def __init__(self):
        super(pconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('bcnt, bmn->bc', (x, A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        return h
    
class pgcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2, temp=1):
        super(pgcn, self).__init__()
        self.nconv = nconv()
        self.temp = temp
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = h[:,:,:,-h.size(3):-self.temp]
        return h

class STPGNN(nn.Module):
    """
    Paper: Spatio-Temporal Pivotal Graph Neural Networks for Trafﬁc Flow Forecasting
    Link: https://ojs.aaai.org/index.php/AAAI/article/view/28707
    Official Code: https://github.com/Kongwy5689/STPGNN?tab=readme-ov-file
    Venue: AAAI 2024
    Task: Spatial-Temporal Forecasting
    """

    def __init__(self, config: STPGNNConfig):
        super(STPGNN, self).__init__()
        num_nodes = int(config.num_nodes)
        dropout = float(config.dropout)
        topk = int(config.topk)
        out_dim = int(config.output_len)
        residual_channels = int(config.residual_channels)
        dilation_channels = int(config.dilation_channels)
        end_channels = int(config.end_channels)
        kernel_size = int(config.kernel_size)
        blocks = int(config.blocks)
        layers = int(config.layers)
        days = int(config.days)
        time_of_day_size = int(config.time_of_day_size)
        dims = int(config.dims)
        order = int(config.order)
        in_dim = int(config.in_dim)
        normalization = str(config.normalization)

        self.input_len = int(config.input_len)
        self.output_len = int(config.output_len)
        self.num_nodes = num_nodes
        skip_channels = 8
        self.alpha = nn.Parameter(torch.tensor(-5.0))  
        self.topk = topk
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.time_of_day_size = time_of_day_size
        self.days = days

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.normal = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.residual_convs_a = nn.ModuleList()
        self.skip_convs_a = nn.ModuleList()
        self.normal_a = nn.ModuleList()
        self.pgconv = nn.ModuleList()

        self.start_conv_a = nn.Conv2d(in_channels=in_dim,
                                      out_channels=1,
                                      kernel_size=(1, 1))
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        receptive_field = 1

        self.supports_len = 1
        self.nodevec_p1 = nn.Parameter(torch.randn(days, dims), requires_grad=True)
        self.nodevec_p2 = nn.Parameter(torch.randn(num_nodes, dims), requires_grad=True)
        self.nodevec_p3 = nn.Parameter(torch.randn(num_nodes, dims), requires_grad=True)
        self.nodevec_pk = nn.Parameter(torch.randn(dims, dims, dims), requires_grad=True)
        self.pivweight = nn.Parameter(torch.randn(num_nodes, num_nodes), requires_grad=True)
  
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))

                self.residual_convs_a.append(nn.Conv2d(in_channels=dilation_channels,
                                                       out_channels=residual_channels,
                                                       kernel_size=(1, 1)))
                
                self.pgconv.append(
                    pgcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len, order=order, temp=new_dilation))
                
                self.gconv.append(
                    gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len, order=order))
                
                if normalization == "batch":
                    self.normal.append(nn.BatchNorm2d(residual_channels))
                    self.normal_a.append(nn.BatchNorm2d(residual_channels))
                elif normalization == "layer":
                    self.normal.append(nn.LayerNorm([residual_channels, num_nodes, 13 - receptive_field - new_dilation + 1]))
                    self.normal_a.append(nn.LayerNorm([residual_channels, num_nodes, 13 - receptive_field - new_dilation + 1]))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.relu = nn.ReLU(inplace=True)

        # Compute total skip channels from temporal widths at each layer.
        # Temporal width starts from max(input_len + 1, receptive_field),
        # because we first left-pad by 1 before entering stacked temporal convs.
        t_cur = max(self.input_len + 1, receptive_field)
        t_sum = 0
        for _ in range(blocks):
            dilation = 1
            for _ in range(layers):
                t_cur = t_cur - dilation * (kernel_size - 1)
                if t_cur <= 0:
                    raise ValueError(
                        f"Invalid temporal width during STPGNN init: t_cur={t_cur}. "
                        f"Check input_len={self.input_len}, blocks={blocks}, layers={layers}, kernel_size={kernel_size}."
                    )
                t_sum += t_cur
                dilation *= 2

        self.end_conv_1 = nn.Conv2d(
            in_channels=skip_channels * t_sum,
            out_channels=end_channels,
            kernel_size=(1, 1),
            bias=True,
        )

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

    def dgconstruct(self, time_embedding, source_embedding, target_embedding, core_embedding):
        adp = torch.einsum('ai, ijk->ajk', time_embedding, core_embedding)
        adp = torch.einsum('bj, ajk->abk', source_embedding, adp)
        adp = torch.einsum('ck, abk->abc', target_embedding, adp)
        adp = F.softmax(F.relu(adp), dim=2)
        return adp
    
    def pivotalconstruct(self, x, adj, k):
        x = x.squeeze(1)
        x = x.sum(dim=0)
        y = x.sum(dim=1).unsqueeze(0)
        adjp = torch.einsum('ij, jk->ik', x[:,:-1], x.transpose(0, 1)[1:,:]) / y
        adjp = adjp * adj
        score = adjp.sum(dim=0) + adjp.sum(dim=1)
        N = x.size(0)
        _, topk_indices = torch.topk(score,k)
        mask = torch.zeros(N, dtype=torch.bool,device=x.device)
        mask[topk_indices] = True
        masked_matrix = adjp * mask.unsqueeze(1) * mask.unsqueeze(0)
        adjp = F.softmax(F.relu(masked_matrix), dim=1)
        return (adjp.unsqueeze(0))

    def forward(self, inputs: torch.Tensor, inputs_timestamps: torch.Tensor = None) -> torch.Tensor:
        # BasicTS compatibility: accept [B, T, N] or [B, T, N, C]
        if inputs.dim() == 3:
            history_data = inputs.unsqueeze(-1)
        elif inputs.dim() == 4:
            history_data = inputs
        else:
            raise ValueError(f"inputs must be [B,T,N] or [B,T,N,C], got {tuple(inputs.shape)}")

        if history_data.shape[1] != self.input_len:
            raise ValueError(f"Expected input_len={self.input_len}, got T={history_data.shape[1]}")
        if history_data.shape[2] != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got N={history_data.shape[2]}")

        x_value = history_data[..., [0]]
        inputs = x_value.permute(0, 3, 2, 1).contiguous()  # [B, T, N, F] -> [B, F, N, T]
        inputs = nn.functional.pad(inputs, (1, 0, 0, 0))
        if inputs_timestamps is not None:
            if inputs_timestamps.dim() >= 3:
                tod = inputs_timestamps[:, -1, 0]
            else:
                raise ValueError(f"inputs_timestamps must be [B,T,C], got {tuple(inputs_timestamps.shape)}")
        elif history_data.shape[-1] > 1:
            tod = history_data[:, -1, 0, 1]
        else:
            tod = torch.zeros(history_data.shape[0], device=history_data.device, dtype=history_data.dtype)
        ind = (tod * self.time_of_day_size).long().to(history_data.device) % self.days
        
        in_len = inputs.size(3)
        num_nodes = inputs.size(2)
        if in_len < self.receptive_field:
            xo = nn.functional.pad(inputs, (self.receptive_field - in_len, 0, 0, 0))
        else:
            xo = inputs
        x = self.start_conv(xo[:, [0]])
        x_a = self.start_conv_a(xo[:, [0]])
        skip = 0
        adj = self.dgconstruct(self.nodevec_p1[ind], self.nodevec_p2, self.nodevec_p3, self.nodevec_pk)
        adj_p = self.pivotalconstruct(x_a, self.pivweight, self.topk)
        supports = [adj]
        supports_a = [adj_p]
    
        for i in range(self.blocks * self.layers):
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x_a = self.pgconv[i](residual, supports_a)
            x = self.gconv[i](x, supports)
            alpha_sigmoid = torch.sigmoid(self.alpha)  
            x = alpha_sigmoid * x_a +  (1 - alpha_sigmoid) * x
            x = x + residual[:, :, :, -x.size(3):]
            s = x
            s = self.skip_convs[i](s)
            if isinstance(skip, int):  # B F N T
                skip = s.transpose(2, 3).reshape([s.shape[0], -1, s.shape[2], 1]).contiguous()
            else:
                skip = torch.cat([s.transpose(2, 3).reshape([s.shape[0], -1, s.shape[2], 1]), skip], dim=1).contiguous()
            x = self.normal[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        # [B, output_len, N, 1] -> [B, output_len, N]
        return x.squeeze(-1)
