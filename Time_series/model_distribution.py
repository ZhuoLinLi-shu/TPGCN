import torch.nn as nn
import torch.nn.functional as F
from ..graph_distribution import graph_constructor
from TPGCN.layer_distribution_mtgnn import *
import numpy as np
from TPGCN.STID_arch import STID
from ..layer_distribution_module import RevIN


class dnconv(nn.Module):
    def __init__(self):
        super(dnconv, self).__init__()

    def forward(self, x, A):
        if len(A.size()) == 2:
            A = A.unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = torch.einsum('nvw, ncwl->ncvl', [A, x])
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn_modify(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn_modify, self).__init__()
        self.nconv = dnconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        x1 = self.nconv(x, support)
        out.append(x1)
        for k in range(2, self.order + 1):
            x2 = self.nconv(x1, support)
            out.append(x2)
            x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gcn_personal(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2, time_length=0):
        super(gcn_personal, self).__init__()
        # self.nconv = nconv()
        self.nconv = dnconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order
        self.mlp_project = nn.ModuleList()
        self.x_project = nn.ModuleList()
        self.mlp_project_1 = nn.ModuleList()
        self.x_project_1 = nn.ModuleList()

        self.query_1 = nn.Conv2d(in_channels=3, out_channels=c_out, kernel_size=(1, 1), bias=True) # 12
        self.query_2 = nn.Conv2d(in_channels=c_out, out_channels=1, kernel_size=(1, 168), bias=True)

        for i in range(self.order):
            self.mlp_project.append(linear(c_out, c_out)) # c_out
            self.mlp_project_1.append(nn.Conv2d(c_out, 1, kernel_size=(1, 1), bias=True))
            self.x_project.append(nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=(1, 1), bias=True))
            self.x_project_1.append(nn.Conv2d(in_channels=c_out, out_channels=1, kernel_size=(1, time_length), bias=True))

    def personal_atten(self, x_driving, mlp_part, query, layer):
        key_mlp = self.mlp_project_1[layer]((self.mlp_project[layer](mlp_part)) + mlp_part)
        key_x = self.x_project_1[layer]((self.x_project[layer](x_driving)) + x_driving)
        key = torch.stack([key_x + query, key_mlp + query], dim=-1).squeeze(-2)
        mid = torch.tanh(key)
        attn = torch.softmax(mid, dim=-1)
        x1 = attn[..., 0].unsqueeze(-1) * x_driving + attn[..., 1].unsqueeze(-1) * mlp_part # .detach()
        return x1

    def forward(self, x, support, mlp_part, pred_time_embed):
        q_mid = (self.query_1(pred_time_embed.transpose(1, 3)))
        query = self.query_2(q_mid)

        out = [x]
        x1_driving = self.nconv(x, support)
        x1 = self.personal_atten(x1_driving, mlp_part, query, layer=0)

        out.append(x1)
        for k in range(2, self.order + 1):
            x2_driving = self.nconv(x1, support)
            x2 = self.personal_atten(x2_driving, mlp_part, query, layer=k-1)
            out.append(x2)
            x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(nn.Module):

    def __init__(self, args, distribute_data=None):
        super(gwnet, self).__init__()
        device = args.device
        num_nodes = args.num_nodes
        dropout = args.dropout

        in_dim = args.in_dim
        residual_channels = args.nhid
        dilation_channels = args.nhid
        skip_channels = args.nhid * 8
        end_channels = args.nhid * 16

        layers = args.layers

        self.dropout = dropout

        self.layers = args.layers
        self.gcn_bool = args.gcn_bool
        self.addaptadj = args.addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.gconv_1 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.nodes = num_nodes

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.seq_length = args.lag
        kernel_size = 7  # 12 #

        dilation_exponential = args.dilation_exponential_
        if dilation_exponential > 1:
            self.receptive_field = int(
                1 + (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        if args.snorm_bool:
            self.sn = nn.ModuleList()
        if args.tnorm_bool:
            self.tn = nn.ModuleList()
        rf_size_i = 1
        new_dilation = 1
        num = 1
        self.snorm_bool = args.snorm_bool
        self.tnorm_bool = args.tnorm_bool

        for j in range(1, layers + 1):
            if dilation_exponential > 1:
                rf_size_j = int(
                    rf_size_i + (kernel_size - 1) * (dilation_exponential ** j - 1) / (dilation_exponential - 1))
            else:
                rf_size_j = rf_size_i + j * (kernel_size - 1)

            self.filter_convs.append(
                dilated_inception(num * residual_channels, dilation_channels, dilation_factor=new_dilation))
            self.gate_convs.append(
                dilated_inception(num * residual_channels, dilation_channels, dilation_factor=new_dilation))

            self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=residual_channels,
                                                 kernel_size=(1, 1)))
            if self.seq_length > self.receptive_field:
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, self.seq_length - rf_size_j + 1)))
            else:
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, self.receptive_field - rf_size_j + 1)))

            if self.gcn_bool:
                self.gconv.append(gcn_personal(dilation_channels, residual_channels, dropout, support_len=1, order=2,
                                               time_length=self.receptive_field - rf_size_j + 1,))

            if self.seq_length > self.receptive_field:
                self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),
                                           elementwise_affine=True))
            else:
                self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),
                                           elementwise_affine=True))
            new_dilation *= dilation_exponential
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=args.output_len,
                                    kernel_size=(1, 1),
                                    bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length),
                                   bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels,
                                   kernel_size=(1, self.seq_length - self.receptive_field + 1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels,
                                   kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1),
                                   bias=True)
        self.idx = torch.arange(self.nodes).to(device)

        self.graph_construct = graph_constructor(args, distribute_data=distribute_data)

        self.MLP_component = STID(num_nodes=self.nodes, input_len=args.lag, output_len=args.output_len,
                                  num_layer=args.MLP_layer, input_dim=args.MLP_indim, node_dim=args.MLP_dim, embed_dim=args.MLP_dim,
                                  temp_dim_tid=args.MLP_dim, temp_dim_diw=args.MLP_dim, if_T_i_D=args.if_T_i_D,
                                  if_D_i_W=args.if_D_i_W, if_node=args.if_node, first_time=args.s_period,
                                  second_time=args.b_period, time_norm=args.time_norm)

        self.mlp_ = nn.Conv2d(in_channels=args.MLP_dim * 4, out_channels=residual_channels, kernel_size=(1, 1), bias=True)
        self.out_norm = LayerNorm((end_channels, num_nodes, 1), elementwise_affine=True)
        self.use_RevIN = args.use_RevIN
        if args.use_RevIN:
            self.revin = RevIN(args.num_nodes)
            self.revin_mlp = RevIN(args.num_nodes)

    def forward(self, input, pred_time_embed=None):
        if self.use_RevIN:
            input = self.revin(input.permute(0, 3, 1, 2), 'norm').permute(0, 2, 3, 1)

        MLP_out, MLP_hidden = self.MLP_component(pred_time_embed)
        MLP_part = MLP_hidden.detach()
        mlp_component = self.mlp_(MLP_part)

        in_len = input.size(3)

        if in_len < self.receptive_field:
            x = nn.functional.pad(input, [self.receptive_field - in_len, 0, 0, 0])
        else:
            x = input

        new_supports = None
        gl_loss = None

        if self.gcn_bool:
            adp, resolution_static, _, gl_loss_from = self.graph_construct(input, mlp_component)
            gl_loss = gl_loss_from
            if self.addaptadj:
                new_supports = adp

        skip = self.skip0(F.dropout(x, self.dropout, training=self.training))
        x = self.start_conv(x)
        # WaveNet layers
        for i in range(self.layers):
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](x)

            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate

            x = F.dropout(x, self.dropout, training=self.training)
            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip

            if self.gcn_bool:
                x = self.gconv[i](x, new_supports, mlp_component, pred_time_embed)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.norm[i](x, self.idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        if self.use_RevIN:
            x = self.revin(x.transpose(2, 3), 'denorm').transpose(2, 3)

        return x, None, MLP_out
