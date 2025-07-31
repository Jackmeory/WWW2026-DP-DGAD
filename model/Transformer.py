import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(CustomTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, raw_src, src_mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, raw_src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return output

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, raw_src, src_mask=None, src_key_padding_mask=None):
        # Q, K from src, V from raw_src
        q = k = src
        v = raw_src

        src2 = self.self_attn(q, k, v, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerBinaryClassifier(nn.Module):
    def __init__(self, config, device, hidden_size=128):
        super(TransformerBinaryClassifier, self).__init__()
        self.device = device
        self.input_size = config.input_dim

        self.encoder_layer = CustomTransformerEncoderLayer(d_model=self.input_size, nhead=config.n_heads,
                                                           dropout=config.drop_out, dim_feedforward=hidden_size)
        self.transformer_encoder = CustomTransformerEncoder(self.encoder_layer, num_layers=config.n_layer)

        self.bn = nn.BatchNorm1d(self.input_size)
        self.dropout = nn.Dropout(config.drop_out)
        # Remove the standard classifier
        # self.classifier = nn.Linear(self.input_size, 1)
        # self.sigmoid = nn.Sigmoid()
        
        # InfProto parameter (controls variance impact)

        
        self.to(device)
        self.config = config
        
        # For tracking running statistics
        self.momentum = config.momentum

    def forward(self, raw_input, GNN_output, mask, normal_prompt=None, abnormal_prompt=None, 
                normal_mean=None, normal_cov=None, abnormal_cov=None, abnormal_mean=None):
        # Encode input
        GNN_output = GNN_output.transpose(0, 1)
        raw_input = raw_input.transpose(0, 1)
        GNN_output = GNN_output.float()
        raw_input = raw_input.float()
        mask = mask.bool()

        transformer_output = self.transformer_encoder(GNN_output, raw_input, src_key_padding_mask=mask)
        transformer_output = transformer_output.transpose(0, 1)
        transformer_output[mask] = 0

        filtered_output = [out[~m] for out, m in zip(transformer_output, mask)]

        averaged_tensors = [tensor.mean(dim=0) for tensor in filtered_output]

        mean_output = torch.stack(averaged_tensors)
        mean_output = self.bn(mean_output)
        
        
        # Create new detached tensors for statistics
        updated_normal_mean = normal_mean.clone().detach()
        updated_abnormal_mean = abnormal_mean.clone().detach()
        updated_normal_cov = normal_cov.clone().detach()
        updated_abnormal_cov = abnormal_cov.clone().detach()
        
        # Only process if we have both normal and abnormal prompts
        if normal_prompt is not None and abnormal_prompt is not None:
            # PART 1: Update statistics (completely detached from computation graph)
            if self.training:
                with torch.no_grad():
                    # Compute batch statistics
                    batch_normal_mean = normal_prompt.mean(0)
                    batch_abnormal_mean = abnormal_prompt.mean(0)
                    
                    # Update means
                    updated_normal_mean = self.momentum * normal_mean + (1 - self.momentum) * batch_normal_mean
                    updated_abnormal_mean = self.momentum * abnormal_mean + (1 - self.momentum) * batch_abnormal_mean
                    
                    # Compute covariances
                    normal_centered = (normal_prompt - normal_mean).unsqueeze(0)
                    abnormal_centered = (abnormal_prompt - abnormal_mean).unsqueeze(0)
                    
                    batch_normal_cov = torch.mm(normal_centered.t(), normal_centered) / max(1, normal_centered.size(0) - 1)
                    batch_abnormal_cov = torch.mm(abnormal_centered.t(), abnormal_centered) / max(1, abnormal_centered.size(0) - 1)
                    
                    # Update covariances
                    updated_normal_cov = self.momentum * normal_cov + (1 - self.momentum) * batch_normal_cov
                    updated_abnormal_cov = self.momentum * abnormal_cov + (1 - self.momentum) * batch_abnormal_cov
            
            # PART 2: Calculate classification scores (with fresh computational graph)
            # Use a fixed value for lambda instead of a parameter
            lambda_val = 1e-3  # Fixed value instead of self.lambda_param
            
            # Calculate scores for each feature vector
            normal_scores_list = []
            abnormal_scores_list = []
            
            for f in mean_output:
                # Create local copies to ensure no in-place operations affect originals
                nm = normal_mean.clone().detach()
                am = abnormal_mean.clone().detach()
                nc = normal_cov.clone().detach()
                ac = abnormal_cov.clone().detach()
                
                # Calculate the dot products
                mu_term_normal = torch.dot(f, nm)
                cov_product = torch.mv(nc, f)  # Matrix-vector product
                cov_term_normal = 0.5 * lambda_val * torch.dot(f, cov_product)
                normal_score = mu_term_normal + cov_term_normal
                
                mu_term_abnormal = torch.dot(f, am)
                cov_product = torch.mv(ac, f)  # Matrix-vector product
                cov_term_abnormal = 0.5 * lambda_val * torch.dot(f, cov_product)
                abnormal_score = mu_term_abnormal + cov_term_abnormal
                
                normal_scores_list.append(normal_score)
                abnormal_scores_list.append(abnormal_score)
            
            # Stack scores into tensors
            normal_scores = torch.stack(normal_scores_list)
            abnormal_scores = torch.stack(abnormal_scores_list)
            
            # Compute final logits
            raw_logits = abnormal_scores - normal_scores
            logits = torch.sigmoid(raw_logits)
        
        # Return results
        return logits, mean_output, updated_normal_mean, updated_normal_cov, updated_abnormal_cov, updated_abnormal_mean


class PrototypeAttention(nn.Module):
    def __init__(self):
        super(PrototypeAttention, self).__init__()
        # Projection layers if dimensions don't match
        
    def forward(self, prototype, subgraph_embedding):
        """
        Args:
            prototype: Fixed prototype of shape (prototype_dim)
            subgraph_embedding: Variable-sized embedding of shape (subgraph_size, embedding_dim)
        Returns:
            Updated prototype representation
        """

        
        # Expand prototype to match subgraph nodes
        prototype = prototype.unsqueeze(0).expand(subgraph_embedding.shape[0], -1)  # (subgraph_size, embedding_dim)
        
        # Compute attention scores
        # Option 1: Simple dot product
        attention_scores = torch.sum(prototype * subgraph_embedding, dim=1)  # (subgraph_size)
        
        # Option 2: MLP-based attention (often better)
        # combined = torch.tanh(query + subgraph_embedding)  # (subgraph_size, embedding_dim)
        # attention_scores = self.attention_layer(combined).squeeze(-1)  # (subgraph_size)
        
        # Normalize attention scores
        attention_weights = F.softmax(attention_scores, dim=0)  # (subgraph_size)
        
        # Apply attention to get context vector
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * subgraph_embedding, dim=0)  # (embedding_dim)
        
        # Option: Combine context with original prototype
        # updated_prototype = prototype + context_vector  # If dimensions match
        
        return context_vector