import torch
from torch import nn
import torch.nn.functional as F


class fc_layer(nn.Module):
    def __init__(self, in_channels, out_channels, need_layer_norm):
        super(fc_layer, self).__init__()
        self.linear_w = nn.Parameter(torch.zeros(size=(in_channels, out_channels)))
        nn.init.xavier_uniform_(self.linear_w.data, gain=1.414)

        self.linear = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=[1, 1], bias=True)
        self.layer_norm = nn.LayerNorm(out_channels)
        self.need_layer_norm = need_layer_norm

    def forward(self, input):
        '''
        input = batch_size, in_channels, nodes, time_step
        output = batch_size, out_channels, nodes, time_step
        '''
        if self.need_layer_norm:
            result = F.leaky_relu(torch.einsum('bani,io->bano ', [input.transpose(1, -1), self.linear_w]))
        else:
            result = F.leaky_relu(torch.einsum('bani,io->bano ', [input.transpose(1, -1), self.linear_w]))
        return result.transpose(1, -1)


class gatedFusion(nn.Module):
    def __init__(self, dim, device):
        super(gatedFusion, self).__init__()
        self.device = device
        self.dim = dim
        self.w = nn.Linear(in_features=dim, out_features=dim)
        self.t = nn.Parameter(torch.zeros(size=(self.dim, self.dim)))
        nn.init.xavier_uniform_(self.t.data, gain=1.414)

        self.w_r = nn.Linear(in_features=dim, out_features=dim)
        self.u_r = nn.Linear(in_features=dim, out_features=dim)

        self.w_h = nn.Linear(in_features=dim, out_features=dim)
        self.w_u = nn.Linear(in_features=dim, out_features=dim)

    def forward(self, batch_size, nodevec, time_node):
        if batch_size == 1 and len(time_node.shape) < 3:
            time_node = time_node.unsqueeze(0)
        node_res = self.w(nodevec) + nodevec
        # node_res = batch_size, nodes, dim
        node_res = node_res.unsqueeze(0).repeat(batch_size, 1, 1)

        time_res = time_node + torch.einsum('bnd, dd->bnd', [time_node, self.t])

        # z = batch_size, nodes, dim
        z = torch.sigmoid(node_res + time_res)
        r = torch.sigmoid(self.w_r(time_node) + self.u_r(nodevec).unsqueeze(0).repeat(batch_size, 1, 1))
        h = torch.tanh(self.w_h(time_node) + r * (self.w_u(nodevec).unsqueeze(0).repeat(batch_size, 1, 1)))
        res = torch.add(z * nodevec, torch.mul(torch.ones(z.size()).to(self.device) - z, h))

        return res


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel * reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel * reduction, channel, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x*y.expand_as(x)


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features,))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features,))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x