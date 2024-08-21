import math
import torch.nn as nn
import models.basicblock as B
import functools
import torch.nn.functional as F
import torch.nn.init as init


from torch import nn
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, '../deep_learning')
from models.constraints import *
import cv2
from models.network_unet import UNet
#from torch_geometric.utils import from_networkx

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EdgeNet(nn.Module):
    def __init__(self, in_features, num_features, ratio=(2, 1)):
        super(EdgeNet, self).__init__()
        num_features_list = [num_features * r for r in ratio]
        # define layers
        layer_list = OrderedDict()
        for l in range(len(num_features_list)):
            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=num_features_list[l-1] if l > 0 else in_features,
                out_channels=num_features_list[l], kernel_size=1, bias=False
            )
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=num_features_list[l])
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()
        # add final similarity kernel
        layer_list['conv_out'] = nn.Conv2d(in_channels=num_features_list[-1],
                                           out_channels=1, kernel_size=1)
        self.sim_network = nn.Sequential(layer_list)

    def forward(self, node_feat): #required ([32, 256])
        node_feat = node_feat.unsqueeze(dim=0)
        num_tasks = node_feat.size(0)
        num_data = node_feat.size(1)
        x_i = node_feat.unsqueeze(2)
        x_j = torch.transpose(x_i, 1, 2)
        x_ij = torch.abs(x_i - x_j)
        x_ij = torch.transpose(x_ij, 1, 3) #shape [1,c,n,n]
        # compute similarity/dissimilarity (batch_size x feat_size x num_samples x num_samples)
        sim_val = torch.sigmoid(self.sim_network(x_ij)).squeeze(1).squeeze(0)
        # normalize affinity matrix
        force_edge_feat = torch.eye(num_data).unsqueeze(0).repeat(num_tasks, 1, 1).to(device)
        edge_feat = sim_val + force_edge_feat
        edge_feat = edge_feat + 1e-6
        edge_feat = edge_feat / torch.sum(edge_feat, dim=1).unsqueeze(1)
        return edge_feat, sim_val #edge_feat torch.Size([1, 32, 32]), sim_val[32,32]

class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, nc, alpha=3, static_feat=None): #static_feat(2,96,4096)
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = nc
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.k = k
        self.dim = dim
        self.alpha = alpha
        #self.static_feat = static_feat

    def forward(self, idx, static_feat):
        if static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))##(n,40)
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))#(4096,4096)
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(device)
        mask.fill_(float('0'))

        s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask.int().float()#(2,1,4096,4096)
        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        return adj

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.transpose(1,2)#b,n,c
        adj = adj.squeeze(1)
        x = F.relu(self.gc1(x, adj))#(b,64,32)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj) #(2,4096,64)
        return F.relu(x)


class GraphConvolution(Module):
    """
    #Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        self.weighted = self.weight.repeat(input.shape[0], 1, 1) #(2,64,32)
        #support = torch.mm(input, self.weighted)
        support = torch.matmul(input, self.weighted) #(b,n,c)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class MSRResNet0(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=16, upscale=4, act_mode='R', upsample_mode='upconv'):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: number of residual blocks
        upscale: up-scale factor
        act_mode: activation function
        upsample_mode: 'upconv' | 'pixelshuffle' | 'convtranspose'
        """
        super(MSRResNet0, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'

        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        m_head = B.conv(in_nc, nc, mode='C')

        m_body = [B.ResBlock(nc, nc, mode='C'+act_mode+'C') for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            m_uper = upsample_block(nc, nc, mode='3'+act_mode)
        else:
            m_uper = [upsample_block(nc, nc, mode='2'+act_mode) for _ in range(n_upscale)]

        H_conv0 = B.conv(nc, nc, mode='C'+act_mode)
        H_conv1 = B.conv(nc, out_nc, bias=False, mode='C')
        m_tail = B.sequential(H_conv0, H_conv1)

        self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)))#
        self.model2 = B.sequential( *m_uper, m_tail)
        self.gc = graph_constructor(nnodes=32*32, k=20,dim=40, nc=nc, static_feat=True) #c
        # To do here!
        #self.gc = EdgeNet(in_features=nc, num_features=128) #cuda memory
        self.gcn = GCN(nfeat=nc, nhid=nc//3, nclass=nc, dropout=0) #c
        self.idx  = torch.arange(32*32).to(device) #c
        self.para_lambda = nn.Parameter(torch.zeros(1)) #c
        self.unet = UNet(1,1)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        """
        :param x: B,C,W,H
        :return:
        """
        mask = self.unet(x)
                
        #difference = self.pool(hr) - x

        weights = self.sigmoid(1000*mask) #+ bias
        #weights = self.softshrink(difference)

        x = x * weights

        m = torch.where(weights >= 0.5, 1, 0)

        #b,c,w,h= x.shape
        #x = x*mask#c
 
        x = self.model(x) #(2,96,64,64)
        
        #h_m = torch.where(mask >= 0.5, 1, 0) #c #we should also think about strong underestimation in LR! >0.01 and < 100?

        
        #noen = torch.count_nonzero(h_m) #c
        #h_m = h_m.view(m.shape[0], m.shape[1], -1) #c #(2,1,4096)
        #h_m = h_m[...,None] #c
        #h_m = h_m.repeat(1, 1, 1, h_m.shape[-2]) #c
        #h_m2 = h_m.transpose(2, 3) #c#(b,1,4096,4096)
        #h = torch.logical_or(h_m, h_m2) #c #(2,1,4096,4096)

        static_feature = x.view(x.shape[0], x.shape[1], -1) #c#(2,96,4096)
        static_feature = static_feature.transpose(1,2).squeeze() #c#(4096,96)
        #_, adj = self.gc(static_feature)
        adj = self.gc(self.idx, static_feat = static_feature) #c  # this one will be removed later, only exist in model_plain for backpropogation, or use fullA during testing

        out = self.gcn(x, adj) #(b,4096,96) #c
        out = out.view(x.shape) #c
        x = self.model2(self.para_lambda*out+x) #c

        #x = self.model2(x) #rc
        ##(1,1,192,192)

        supervised_nodes = torch.kron(m, torch.ones((3, 3)).to('cuda')) #c
        mask = mask.to(torch.float32)

        return x, supervised_nodes, mask #c
# --------------------------------------------
# modified SRResNet v0.1
# https://github.com/xinntao/ESRGAN
# --------------------------------------------
class MSRResNet1(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=16, upscale=4, act_mode='R', upsample_mode='upconv'):
        super(MSRResNet1, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nc, 3, 1, 1, bias=True)
        basic_block = functools.partial(ResidualBlock_noBN, nc=nc)
        self.recon_trunk = make_layer(basic_block, nb)

        # upsampling
        if self.upscale == 2:
            self.upconv1 = nn.Conv2d(nc, nc * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2d(nc, nc * 9, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(nc, nc * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nc, nc * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nc, nc, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nc, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last], 0.1)
        if self.upscale == 4:
            initialize_weights(self.upconv2, 0.1)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        out = self.recon_trunk(fea)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.HRconv(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nc=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nc, nc, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nc, nc, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out