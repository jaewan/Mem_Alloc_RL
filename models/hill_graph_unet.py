import torch
import torch.nn.functional as F
from torch_sparse import spspmm
from torch_geometric.nn import TopKPooling, GCNConv, GATConv
from torch_geometric.utils import add_self_loops, remove_self_loops, sort_edge_index, to_undirected
from torch_geometric.utils.repeat import repeat
def identity(x):
    return x

class BidirectionalGraphConv(torch.nn.Module):
    def __init__(self, in_c, out_c, n_attention_heads=0, concat=True):
        super(BidirectionalGraphConv, self).__init__()
        self.use_gat = n_attention_heads > 0
        self.concat = concat
        if concat:
            # divide each direction conv out_x by 2, to preserve out_c channels at total
            out_c = out_c // 2

        if self.use_gat:
            self.inbound_conv = GATConv(in_c, out_c // n_attention_heads, heads=n_attention_heads)
            self.outbound_conv = GATConv(in_c, out_c // n_attention_heads, heads=n_attention_heads)
        else:
            self.inbound_conv = GCNConv(in_c, out_c, improved=True)
            self.outbound_conv = GCNConv(in_c, out_c, improved=True)

    def forward(self, x, edge_index, edge_weight=None):
        # create inverted edges for the backward conv:
        edge_index_inv = edge_index.flip(0) # todo swap rows of edge_index to create a copy of inverted edges
        if self.use_gat:
            inbound_x = self.inbound_conv(x, edge_index)
            outbound_x = self.outbound_conv(x, edge_index_inv)
        else:
            inbound_x = self.inbound_conv(x, edge_index, edge_weight)
            outbound_x = self.outbound_conv(x, edge_index_inv, edge_weight)
        if self.concat:
            # todo concatenate the features from both convs
            x = torch.cat([inbound_x, outbound_x], dim=1)
        else:
            # todo sum the features from both convs
            x = inbound_x + outbound_x
        return x

    def reset_parameters(self):
        self.inbound_conv.reset_parameters()
        self.outbound_conv.reset_parameters()

class GCNBlock(torch.nn.Module):
    def __init__(self, in_channels,
                 hidden_channels,
                 out_channels,
                 depth=1,
                 inter_residual_connection=False,
                 global_residual_connection=False,
                 out_act=identity,
                 inter_act=identity):
        """
        GCNBlock is a chain of GCNConv operations,
        interleaved by activations and residual connections,
        as proposed in GCN paper.
        :param in_channels: (int) Input features dimension
        :param hidden_channels: (int) Features dimension for all intermediate GCN layers in the block
        :param out_channels: (int) Output features dimension
        :param depth: (int) Number of GCNConv layers.
        :param inter_residual_connection: Add the input features of each GCNConv layer to its activation output. (default: False)
        :param global_residual_connection: Add the block input features to the block output before applying out_act. (default: False)
        :param out_act: Activation to apply to the blcck output. (default: identity)
        :param inter_act: Activation for each intermediate GCNConv layer. (default: identity)
        """
        super(GCNBlock, self).__init__()
        self.inter_residual_connection = inter_residual_connection
        self.global_residual_connection = global_residual_connection
        self.depth = depth
        self.layers = torch.nn.ModuleList()
        if depth > 1:
            self.layers.append(GCNConv(in_channels, hidden_channels, improved=True))
            for l in range(1, self.depth - 1):
                self.layers.append(GCNConv(hidden_channels, hidden_channels, improved=True))
            self.layers.append(GCNConv(hidden_channels, out_channels, improved=True))
        else:
            self.layers.append(GCNConv(in_channels, out_channels, improved=True))
        self.inter_act = inter_act
        self.out_act = out_act

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self,x, edge_index, edge_weight=None):
        global_identity = x
        out = x
        for layer in self.layers:
            inter_identity = out
            out = self.inter_act(layer(out, edge_index, edge_weight))
            # Add residual connection to the activation as in the paper
            if self.inter_residual_connection:
                out += inter_identity
        # Add global residual connection before the activation as in ResNet basic blocks.
        if self.global_residual_connection:
            out += global_identity
        out = self.out_act(out)
        return out

class HillGraphUNet(torch.nn.Module):
    r""" Improved version of the Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.
    The main improvements:
        - deeper model - multiple GCNConv layers per stage,
        - allow deep message propagation through network.
        - residual connections as proposed in the GCNConv paper.
    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)

        num_pre_gcn (int): The number of GCNConv layers to apply before the first pooling
        num_inter_gcn (int): The number of GCNConv layers to apply between each pooling and
                             Unpooling operations.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, depth,
                 params={},
                 pool_ratios=0.5, sum_res=True, act=F.relu):
        super(HillGraphUNet, self).__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.undirected_graphs = False;#params['undirected_graphs']
        self.n_attention_heads = 0#params['n_attention_heads']
        #self.attention_concat = params['attention_concat']
        self.pool_ratios = repeat(pool_ratios, depth)
        self.act = act
        self.sum_res = sum_res
        self.use_batchnorm = params.get('use_batchnorm', True)
        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_bn = torch.nn.ModuleList() # batch norm after each GCN block

        # select between GCNConv and GATConv (graph attention) and BidirectionalGraphConv
        '''
        if params['bidirectional_graph_conv']:
            graph_conv = lambda in_c, out_c: BidirectionalGraphConv(in_c, out_c, n_attention_heads=self.n_attention_heads, concat=True)
        '''
        if self.n_attention_heads == 0:
            graph_conv = lambda in_c, out_c: GCNBlock(in_c, channels, out_c)
        else:
            # we divide out_c by n_attention_heads because GATConv output is
            # by default the concatenation of all heads.
            # to get out_c channels at total,
            # we specify the attention heads with (out_c / n_attention_heads) output channels each
            # the user must verify that n_attention_heads divides out_c. n_heads Outputs of size outc//n_heads concatenated
            # one to each other >>> output of size out_c
            graph_conv = lambda in_c, out_c: GATConv(in_c, out_c // self.n_attention_heads, heads=self.n_attention_heads)


        self.down_convs.append(graph_conv(in_channels, channels))
        self.down_bn.append(torch.nn.BatchNorm1d(channels))

        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_convs.append(graph_conv(channels, channels))
            self.down_bn.append(torch.nn.BatchNorm1d(channels))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        self.up_bn = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(graph_conv(in_channels, channels))
            self.up_bn.append(torch.nn.BatchNorm1d(channels))

        self.up_convs.append(graph_conv(in_channels, out_channels))
        self.up_bn.append(torch.nn.BatchNorm1d(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()
        for bn in self.down_bn:
            bn.reset_parameters()
        for bn in self.up_bn:
            bn.reset_parameters()

    def forward(self, x, edge_index, batch=None):
        """"""
        # all graphs are directed by default.
        # in order to perform graph convolutions on undirected edges
        # we convert all graphs to undirected.
        # this might enhance information propagation.
        temp = x
        if self.undirected_graphs:
            # for all (i,j) in edge_index include also (j,i)
            edge_index = to_undirected(edge_index)

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))
        if self.n_attention_heads == 0:
            x = x.type(torch.FloatTensor)
            x = self.down_convs[0](x, edge_index, edge_weight)
        else:
            x = x.type(torch.FloatTensor)
            x = self.down_convs[0](x, edge_index)

        x = self.act(x)
        if self.use_batchnorm:
            x = self.down_bn[0](x)
        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)

            if self.n_attention_heads == 0:
                x = self.down_convs[i](x, edge_index, edge_weight)
            else:
                x = self.down_convs[i](x, edge_index)

            x = self.act(x)
            if self.use_batchnorm:
                x = self.down_bn[i](x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)
            if self.n_attention_heads == 0:
                x = self.up_convs[i](x, edge_index, edge_weight)
            else:
                x = self.up_convs[i](x, edge_index)

            x = self.act(x) if i < self.depth - 1 else x
            if self.use_batchnorm:
                x = self.up_bn[i](x)

        return x

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def __repr__(self):
        return '{}({}, {}, {}, depth={}, pool_ratios={})'.format(
            self.__class__.__name__, self.in_channels, self.hidden_channels,
            self.out_channels, self.depth, self.pool_ratios)

