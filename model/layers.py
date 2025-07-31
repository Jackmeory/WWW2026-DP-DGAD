import math
import sys

import numpy as np
import torch
import time

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    def __init__(self, in_features_v, out_features_v, in_features_e, out_features_e, bias=True, node_layer=True):
        super(GraphConvolution, self).__init__()
        self.in_features_e = in_features_e
        self.out_features_e = out_features_e
        self.in_features_v = in_features_v
        self.out_features_v = out_features_v

        if node_layer:
            self.node_layer = True
            self.weight = Parameter(torch.FloatTensor(in_features_v, out_features_v))
            self.p = Parameter(torch.from_numpy(np.random.normal(size=(1, in_features_e))).float())
            if bias:
                self.bias = Parameter(torch.FloatTensor(out_features_v))
            else:
                self.register_parameter('bias', None)
        else:
            self.node_layer = False
            self.weight = Parameter(torch.FloatTensor(in_features_e, out_features_e))
            self.p = Parameter(torch.from_numpy(np.random.normal(size=(1, in_features_v))).float())
            if bias:
                self.bias = Parameter(torch.FloatTensor(out_features_e))
            else:
                self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, H_v, H_e, adj_e, adj_v, T):
        # Convert inputs to dense if they're sparse
        if isinstance(T, torch.Tensor) and T.is_sparse:
            T_dense = T.to_dense()
        else:
            T_dense = T
            
        if isinstance(adj_v, torch.Tensor) and adj_v.is_sparse:
            adj_v_dense = adj_v.to_dense()
        else:
            adj_v_dense = adj_v
            
        if isinstance(adj_e, torch.Tensor) and adj_e.is_sparse:
            adj_e_dense = adj_e.to_dense()
        else:
            adj_e_dense = adj_e
        
        if self.node_layer:
            # Use dense operations consistently
            diag_values = (H_e @ self.p.t()).t()[0]
            diag_matrix = torch.diag(diag_values)
            multiplier1 = torch.mm(T_dense, torch.mm(diag_matrix, T_dense.t()))
            
            mask1 = torch.eye(multiplier1.shape[0], device=multiplier1.device)
            M1 = mask1 * torch.ones(multiplier1.shape[0], device=multiplier1.device) + (1. - mask1) * multiplier1
            
            adjusted_A = torch.mul(M1, adj_v_dense)
            output = torch.mm(adjusted_A, torch.mm(H_v, self.weight))
            
            if self.bias is not None:
                ret = output + self.bias
            return ret, H_e

        else:
            # Use dense operations consistently
            diag_values = (H_v @ self.p.t()).t()[0]
            diag_matrix = torch.diag(diag_values)
            multiplier2 = torch.mm(T_dense.t(), torch.mm(diag_matrix, T_dense))
            
            mask2 = torch.eye(multiplier2.shape[0], device=multiplier2.device)
            M3 = mask2 * torch.ones(multiplier2.shape[0], device=multiplier2.device) + (1. - mask2) * multiplier2
            
            adjusted_A = torch.mul(M3, adj_e_dense)
            # Perform normalization
            normalized_adjusted_A = adjusted_A / adjusted_A.max(0, keepdim=True)[0].clamp(min=1e-10)
            
            output = torch.mm(normalized_adjusted_A, torch.mm(H_e, self.weight))
            if self.bias is not None:
                ret = output + self.bias
            return H_v, ret

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features_v) + ',' + str(self.in_features_e) + ' -> ' \
               + str(self.out_features_v) + ',' + str(self.out_features_e) + ')'