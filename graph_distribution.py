import torch
from torch import nn
from .layer_distribution_module import *
from torch.nn.utils import weight_norm
from .layer_distribution_mtgnn import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(logits, temperature, eps=1e-10):
    sample = sample_gumbel(logits.size(), eps=eps)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False, eps=1e-10):
    y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps)

    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape).to(device)
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y, y_soft


class dynamic_graph(nn.Module):
    def __init__(self, args, distribute_data=None):
        super(dynamic_graph, self).__init__()

        self.head_dim = args.headdim
        self.heads = args.headnum
        self.node_dim = args.embed_dim
        self.D = self.heads * self.head_dim  # node_dim #

        self.dropout = args.dropout_ingc

        self.nodes = args.num_nodes

        self.query = fc_layer(in_channels=self.node_dim, out_channels=self.D, need_layer_norm=False)
        self.key = fc_layer(in_channels=self.node_dim, out_channels=self.D, need_layer_norm=False)

        self.mlp = nn.Conv2d(in_channels=self.heads, out_channels=self.heads, kernel_size=(1, 1), bias=True)

        self.bn = nn.LayerNorm(self.node_dim)
        self.bn1 = nn.LayerNorm(self.node_dim)
        self.w = nn.Parameter(torch.zeros(size=(self.nodes, self.node_dim)))
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        self.attn_static = nn.LayerNorm(self.nodes)
        self.skip_norm = nn.LayerNorm(self.nodes)
        self.attn_norm = nn.LayerNorm(self.nodes)

        self.static_inf_norm = nn.LayerNorm(self.nodes)

        self.temperature = args.temperature

        self.channel_attn = SELayer(channel=3)
        self.channel_mlp = nn.Conv2d(in_channels=3, out_channels=args.nhid, kernel_size=(1, 1))
        self.channel_mlp1 = nn.Conv2d(in_channels=args.nhid, out_channels=2, kernel_size=(1, 1))
        self.dynamic_inf_norm = nn.LayerNorm(self.nodes)
        self.train_feas = distribute_data

    def forward(self, nodevec_fuse, input_gc, nodevec_static, input):
        batch_size, nodes, node_dim = input_gc.shape[0], self.nodes, self.node_dim

        node_orginal = nodevec_fuse
        nodevec_fuse = self.bn(nodevec_fuse)
        # Edge weights are inferred under homogeneous information from learned node embeddings
        static_graph_inf = self.static_inf_norm(torch.mm(nodevec_static, nodevec_static.transpose(1, 0)))

        # Edge weights are inferred under homogeneous information from dynamic node-level input
        dynamic_input_matrix = self.dynamic_inf_norm(
            torch.einsum('bnd, bdm -> bnm', input_gc, input_gc.transpose(1, 2)))

        # Edge weights are inferred under heterogeneous information
        nodevec = torch.einsum('bnd, nl -> bnl', nodevec_fuse, self.w) + nodevec_fuse
        skip_atten = torch.einsum('bnd,bdm->bnm', nodevec, nodevec.transpose(-1, -2))
        skip_atten = self.skip_norm(skip_atten)

        nodevec_fuse = nodevec_fuse.unsqueeze(1).transpose(1, -1)

        # Multi-head mechanism
        query = self.query(nodevec_fuse)
        key = self.key(nodevec_fuse)
        key = key.squeeze(-1).contiguous().view(batch_size, self.heads, self.head_dim, nodes)
        query = query.squeeze(-1).contiguous().view(batch_size, self.heads, self.head_dim, nodes).transpose(-1, -2)
        attention = torch.einsum('bhnd, bhdu-> bhnu', query, key)
        attention /= (self.head_dim ** 0.5)
        attention = F.dropout(attention, self.dropout, training=self.training)
        static_graph_inf = static_graph_inf.unsqueeze(0).repeat(batch_size, 1, 1)
        edgeHe = self.attn_norm(torch.sum(attention, dim=1)) + skip_atten

        weight_raw = torch.stack([edgeHe.unsqueeze(1), static_graph_inf.unsqueeze(1),
                               dynamic_input_matrix.unsqueeze(1)], dim=1).squeeze()
        # Channel Attention
        adj_channel = self.channel_attn(weight_raw) + weight_raw
        adj_edge = F.dropout(self.channel_mlp1(torch.relu(self.channel_mlp(adj_channel))), self.dropout,
                              training=self.training)
        adj_edge = adj_edge.permute(0, 2, 3, 1).reshape(batch_size, nodes * nodes, 2)
        # Get discrete graph structures
        adj, _ = gumbel_softmax(adj_edge, temperature=self.temperature, hard=True)
        prop_adj = adj[..., 0].clone().reshape(batch_size, nodes, nodes)
        mask = torch.eye(nodes, nodes).bool().to(device).unsqueeze(0).repeat(batch_size, 1, 1)
        prop_adj.masked_fill_(mask, 0)

        # Normalized graph matrices
        prop_adj = self._calculate_random_walk_matrix(prop_adj)
        gl_loss = None

        return prop_adj, static_graph_inf, node_orginal, gl_loss,

    def _calculate_random_walk_matrix(self, adj_mx):
        batch_size = adj_mx.shape[0]
        adj_eye = torch.eye(int(adj_mx.shape[1])).to(device).unsqueeze(0).repeat(batch_size, 1, 1)
        adj_mx = adj_mx + adj_eye

        d = torch.sum(adj_mx, -1)
        d_inv = 1. / d

        d_inv = torch.where(torch.isinf(d_inv), torch.zeros(d_inv.shape).to(device), d_inv)

        d_mat = []
        for i in range(adj_mx.shape[0]):
            d_mat.append(torch.diag(d_inv[i]))

        d_mat_adj = torch.cat(d_mat, dim=0).reshape(-1, self.nodes, self.nodes)
        random_walk_mx = torch.einsum('bnn, bnm -> bnm', d_mat_adj, adj_mx)
        return random_walk_mx


class graph_constructor(nn.Module):
    def __init__(self, args, cout=16, is_add1=True, distribute_data=None, ):
        super(graph_constructor, self).__init__()
        self.nums = 1

        nodes = args.num_nodes
        time_step = args.lag
        in_dim = args.in_dim

        self.embed1 = nn.Embedding(nodes, args.embed_dim)

        self.out_channel = cout
        self.device = args.device

        self.nodes = nodes
        self.node_dim = args.embed_dim

        self.time_length = time_step
        self.time_norm = nn.LayerNorm(args.embed_dim)
        self.up_channel = nn.Conv2d(in_dim, out_channels=self.out_channel, kernel_size=(1, 1))

        self.down_channel = nn.Conv2d(self.out_channel, args.embed_dim, kernel_size=(1, self.time_length))
        self.trans_Merge_line = nn.Conv2d(in_dim * self.nums, args.embed_dim, kernel_size=(1, self.time_length))
        self.aware_t_adj = dynamic_graph(args, distribute_data=distribute_data)

        self.gate_Fusion = gatedFusion(args.embed_dim, device)

    def forward(self, raw_input, mlp_component):

        batch_size, nodes = raw_input.shape[0], self.nodes
        idx = torch.arange(self.nodes).to(self.device)
        nodevec = self.embed1(idx)

        input_feature = self.time_norm(self.trans_Merge_line(raw_input).squeeze(-1).transpose(1, 2))

        node_fuse = self.gate_Fusion(batch_size, nodevec, input_feature) + nodevec

        aware_adj = self.aware_t_adj(node_fuse, input_feature, nodevec, raw_input)
        return aware_adj