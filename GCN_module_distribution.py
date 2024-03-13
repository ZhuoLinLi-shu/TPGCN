from torch import nn
import torch
import math
import torch.nn.functional as F

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', [x, A])
        return x.contiguous()


class dnconv(nn.Module):
    def __init__(self):
        super(dnconv, self).__init__()

    def forward(self, x, A):
        if len(A.size()) == 2:
            A = A.unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = torch.einsum('nvw, ncwl->ncvl', [A, x])
        return x.contiguous()


class dy_nconv(nn.Module):
    def __init__(self):
        super(dy_nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,nvd->ncdl', [x, A])
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class nconv_GCN(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(nconv_GCN, self).__init__()
        self.in_feature = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_patameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, A):
        support = torch.mm(x, self.weight)
        output = torch.spmm(A, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class weight_node(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim=10):
        super(weight_node, self).__init__()
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, in_channels, out_channels))
        nn.init.xavier_uniform_(self.weights_pool.data, gain=1.414)

    def forward(self, node_embeddings):
        if len(node_embeddings.shape) < 3:
            node_embeddings = node_embeddings.unsqueeze(0).repeat(64, 1, 1)
        result = torch.einsum('bnd,dio->bnio', node_embeddings, self.weights_pool)
        result = result.permute(0, 2, 3, 1)
        return result


class weight_static(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim=10):
        super(weight_static, self).__init__()
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, in_channels, out_channels))
        nn.init.xavier_uniform_(self.weights_pool.data, gain=1.414)

    def forward(self, node_embeddings):
        result = torch.einsum('nd,dio->nio', node_embeddings, self.weights_pool)
        return result