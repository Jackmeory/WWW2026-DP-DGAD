import sys

import torch
import torch.nn.functional as F
#from torch_geometric.nn import GCNConv
#from torch_geometric.data import Data
import torch.optim as optim
import torch.nn as nn
import time


class DGG(torch.nn.Module):
    def __init__(self, GNN, Transformer,transformer_d,max_hop=1, max_dist=2, max_nodes=41, embedding_dim=128):
        super(DGG, self).__init__()
        self.GNN = GNN
        self.Transformer = Transformer
        self.Transformer_d = transformer_d
        self.act = nn.PReLU()
        # Initialize embedding matrices
        self.Ehe = nn.Parameter(torch.randn(max_hop+1, embedding_dim))
        self.Ede = nn.Parameter(torch.randn(max_dist+1, embedding_dim))
        self.Eft = nn.Parameter(torch.randn(max_nodes, embedding_dim))
        self.Ece = nn.Parameter(torch.randn(4, embedding_dim))


    def forward(self, H_v, H_e, H_e_pad, adj_e, adj_v, T, mask, ra, normal_prompt_raw, abnormal_prompt_raw,normal_mean, normal_cov, abnormal_cov,abnormal_mean):
    # Fuse embeddings (Add strategy)

        #normal_prompt = self.act(self.Transformer_d(normal_prompt_raw, ra))
        #abnormal_prompt = self.act(self.Transformer_d(abnormal_prompt_raw, ra))

        normal_prompt = normal_prompt_raw
        abnormal_prompt = abnormal_prompt_raw
  


        output = self.GNN(H_v, H_e, adj_e, adj_v, T)
        pad_output = torch.zeros((H_e_pad.shape[0], H_e_pad.shape[1], H_e_pad.shape[2])).to('cuda')
        for i in range(H_e_pad.shape[0]):
            length = output[i].shape[0]
            pad_output[i, :length, :] = output[i]


        logits, output, normal_mean, normal_cov, abnormal_cov,abnormal_mean= self.Transformer(H_e_pad, pad_output, mask, normal_prompt, abnormal_prompt,normal_mean, normal_cov, abnormal_cov,abnormal_mean)

        #print(f"Total time: {total_time:.4f}s")
        #print(f"- Node embeddings: {node_time:.4f}s ({node_time/total_time*100:.1f}%)")
        #print(f"- Edge embeddings: {edge_time:.4f}s ({edge_time/total_time*100:.1f}%)")
        #print(f"- GNN: {gnn_time:.4f}s ({gnn_time/total_time*100:.1f}%)")
        #print(f"- Transformer: {transformer_time:.4f}s ({transformer_time/total_time*100:.1f}%)")
        return logits, output, normal_prompt, abnormal_prompt, normal_mean, normal_cov, abnormal_cov,abnormal_mean
