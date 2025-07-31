import sys
import torch
import torch.nn.functional as F
import datasets as dataset
import torch.utils.data
import sklearn
from scipy.stats import rankdata, iqr, trim_mean
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, average_precision_score
import numpy as np
import torch.multiprocessing as mp
import pickle
import csv
from model.CensNet import CensNet
from model.DGG import DGG
from model.LSTM import LSTMBinaryClassifier
from option_source_train import args
from utils import EarlyStopMonitor, logger_config
from tqdm import tqdm
import datetime, os
from model.Transformer import TransformerBinaryClassifier, PrototypeAttention
import random
from collections import defaultdict

import warnings


import warnings
warnings.filterwarnings("ignore")

def main():
    def criterion(logits, labels):
        labels = labels
        loss_classify = F.binary_cross_entropy_with_logits(
            logits, labels, reduction='none')
        loss_classify = torch.mean(loss_classify)
        return loss_classify

    def process_batch(batch_sample, model, device,normal_mean, normal_cov, abnormal_cov,abnormal_mean):
        """Common batch processing for both training and evaluation"""
        input_nodes_feature = batch_sample['input_nodes_feature']
        input_edges_feature = batch_sample['input_edges_feature']
        input_edges_pad = batch_sample['input_edges_pad']
        labels = batch_sample['labels']
        labels = torch.tensor(labels)
        Tmats = batch_sample['Tmats']
        adjs = batch_sample['adjs']
        eadjs = batch_sample['eadjs']
        mask_edge = batch_sample['mask_edge']

        input_nodes_feature = [tensor.to(device) for tensor in input_nodes_feature]
        input_edges_feature = [tensor.to(device) for tensor in input_edges_feature]
        input_edges_pad = input_edges_pad.to(device)
        mask_edge = mask_edge.to(device)
        Tmats = [tensor.to(device) for tensor in Tmats]
        adjs = [tensor.to(device) for tensor in adjs]
        eadjs = [tensor.to(device) for tensor in eadjs]
        rank = batch_sample['ra']



        # Get prototypes from buffer if available
        if len(prototype_buffer) > 0 and use_buffer:
            # Find most suitable prototype pair
            if current_dataset_idx > 0:  # Not the first dataset
                # Use the pre-calculated best prototype based on relevance
                best_pair_idx = best_prototype_idx
                normal_prompt_raw, abnormal_prompt_raw = prototype_buffer[best_pair_idx][0].to(device), prototype_buffer[best_pair_idx][1].to(device)
            else:
                # For first dataset, just use the most different pair
                best_pair_idx = 0  # Since buffer is sorted by difference
                normal_prompt_raw, abnormal_prompt_raw = prototype_buffer[best_pair_idx][0].to(device), prototype_buffer[best_pair_idx][1].to(device)
        else:
            # Initialize with random if buffer is empty
            normal_prompt_raw = torch.randn(args.input_dim).to(device)
            abnormal_prompt_raw = torch.randn(args.input_dim).to(device)

        logits, output, normal_prompt, abnormal_prompt, nn, nc, ac,an = model(
            input_nodes_feature,
            input_edges_feature,
            input_edges_pad,
            eadjs,
            adjs,
            Tmats,
            mask_edge,

            rank,
            normal_prompt_raw,
            abnormal_prompt_raw,
            normal_mean.to(device), normal_cov.to(device), abnormal_cov.to(device),abnormal_mean.to(device)
        )
        y = labels.to(device)
        y = y.to(torch.float32)
        
        return y, logits, output, normal_prompt, abnormal_prompt, nn, nc, ac,an

    def eval_model(data_loader, model, config, device, normal_mean, normal_cov, abnormal_cov,abnormal_mean):
        """Evaluate model on a dataset"""
        m_loss, m_pred, m_label = np.array([]), np.array([]), np.array([])
        with torch.no_grad():
            model.eval()
            for batch_sample in data_loader:
                y, logits, _, _, _ ,_,_,_,_= process_batch(batch_sample, model, device, normal_mean, normal_cov, abnormal_cov,abnormal_mean)
                c_loss = np.array([criterion(logits, y).cpu()])
                
                pred_score = logits.cpu().numpy().flatten()
                y_np = y.cpu().numpy().flatten()
                
                m_loss = np.concatenate((m_loss, c_loss))
                m_pred = np.concatenate((m_pred, pred_score))
                m_label = np.concatenate((m_label, y_np))
                
            auc_roc = roc_auc_score(m_label, m_pred)
            avg_precision = average_precision_score(m_label, m_pred)
        return np.mean(m_loss), auc_roc, avg_precision

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    def calculate_dataset_embeddings(dataset_loader, model, device, prototype_buffer):
        """Calculate embeddings for the entire dataset"""
        all_embeddings = []
        all_labels = []
        normal_embeddings = []
        abnormal_embeddings = []
        
        model.eval()
        print("Calculating dataset embeddings...")
        
        with torch.no_grad():
            # Use a fixed prototype for consistent embeddings
            if len(prototype_buffer) > 0:
                normal_prompt_raw = prototype_buffer[0][0].to(device)
                abnormal_prompt_raw = prototype_buffer[0][1].to(device)
            else:
                # Initialize random prototypes if buffer is empty
                normal_prompt_raw = torch.randn(args.input_dim).to(device)
                abnormal_prompt_raw = torch.randn(args.input_dim).to(device)
            
            for batch_sample in tqdm(dataset_loader, desc="Generating embeddings"):
                # Extract features
                input_nodes_feature = batch_sample['input_nodes_feature']
                input_edges_feature = batch_sample['input_edges_feature']
                input_edges_pad = batch_sample['input_edges_pad']
                labels = batch_sample['labels']
                labels = torch.tensor(labels).to(device)
                Tmats = [tensor.to(device) for tensor in batch_sample['Tmats']]
                adjs = [tensor.to(device) for tensor in batch_sample['adjs']]
                eadjs = [tensor.to(device) for tensor in batch_sample['eadjs']]
                mask_edge = batch_sample['mask_edge']

                input_nodes_feature = [tensor.to(device) for tensor in input_nodes_feature]
                input_edges_feature = [tensor.to(device) for tensor in input_edges_feature]
                input_edges_pad = input_edges_pad.to(device)
                mask_edge = mask_edge.to(device)
                Tmats = [tensor.to(device) for tensor in Tmats]
                adjs = [tensor.to(device) for tensor in adjs]
                eadjs = [tensor.to(device) for tensor in eadjs]
                rank = batch_sample['ra']
                
                # Get batch embeddings
                _, output, _, _,_,_,_,_ = model(
                    input_nodes_feature,
                    input_edges_feature,
                    input_edges_pad,
                    eadjs,
                    adjs,
                    Tmats,
                    mask_edge,

                    rank,

                    normal_prompt_raw,
                    abnormal_prompt_raw,
                    normal_mean, normal_cov, abnormal_cov,abnormal_mean
                )
                
                # Store embeddings and labels
                all_embeddings.append(output.cpu())
                all_labels.append(labels.cpu())
                
                # Separate normal and abnormal embeddings
                normal_indices = torch.where(labels==1)[0]
                abnormal_indices = torch.where(labels==0)[0]
                
                if len(normal_indices) > 0:
                    normal_embeddings.append(output[normal_indices].cpu())
                
                if len(abnormal_indices) > 0:
                    abnormal_embeddings.append(output[abnormal_indices].cpu())
        
        # Concatenate all results
        all_embeddings = torch.cat(all_embeddings, dim=0) if all_embeddings else torch.tensor([])
        all_labels = torch.cat(all_labels, dim=0) if all_labels else torch.tensor([])
        normal_embeddings = torch.cat(normal_embeddings, dim=0) if normal_embeddings else torch.tensor([])
        abnormal_embeddings = torch.cat(abnormal_embeddings, dim=0) if abnormal_embeddings else torch.tensor([])
        
        return {
            'all_embeddings': all_embeddings,
            'all_labels': all_labels,
            'normal_embeddings': normal_embeddings,
            'abnormal_embeddings': abnormal_embeddings
        }
    
    def calculate_prototype_relevance_scores(dataset_embeddings, prototype_buffer, device):
        """Calculate relevance scores for all prototypes using pre-calculated embeddings"""
        normal_embeddings = dataset_embeddings['normal_embeddings']
        abnormal_embeddings = dataset_embeddings['abnormal_embeddings']
        
        # If either normal or abnormal embeddings are empty, can't calculate proper relevance
        if len(normal_embeddings) == 0 or len(abnormal_embeddings) == 0:
            return [0.5] * len(prototype_buffer)
        
        # Move data to device for faster computation
        normal_embeddings = normal_embeddings.to(device)
        abnormal_embeddings = abnormal_embeddings.to(device)
        
        # Stack all prototypes
        n_protos = torch.stack([n_proto for n_proto, _, _ in prototype_buffer]).to(device)
        a_protos = torch.stack([a_proto for _, a_proto, _ in prototype_buffer]).to(device)
        
        # Calculate distances between all prototypes and all embeddings (vectorized)
        normal_dists = torch.cdist(n_protos, normal_embeddings)  # shape: [num_protos, num_normal]
        abnormal_dists = torch.cdist(a_protos, abnormal_embeddings)  # shape: [num_protos, num_abnormal]
        
        # Calculate relevance scores (negative mean distance)
        normal_relevances = -torch.mean(normal_dists, dim=1)  # shape: [num_protos]
        abnormal_relevances = -torch.mean(abnormal_dists, dim=1)  # shape: [num_protos]
        
        # Combined relevance
        relevance_scores = ((normal_relevances + abnormal_relevances) / 2).cpu().numpy()
        
        # Normalize to [0,1] range
        min_score = np.min(relevance_scores) if len(relevance_scores) > 0 else 0
        max_score = np.max(relevance_scores) if len(relevance_scores) > 0 else 1
        
        if max_score > min_score:
            relevance_scores = (relevance_scores - min_score) / (max_score - min_score)
        else:
            relevance_scores = np.array([0.5] * len(relevance_scores))
        
        return relevance_scores.tolist()
    

    
    def save_model(model, optimizer, dataset_name, prototype_buffer, prototype_relevance_scores, 
                normal_mean, normal_cov, abnormal_cov, abnormal_mean, 
                save_dir="/home/jz/AAAI2025DGG-main/save_model"):
        """Save model and related information including distribution parameters"""
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{dataset_name}_best_model.pt")
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'prototype_buffer': prototype_buffer,
            'prototype_relevance_scores': prototype_relevance_scores,
            'dataset_name': dataset_name,
            'normal_mean': normal_mean.cpu(),  # Move to CPU before saving
            'normal_cov': normal_cov.cpu(),
            'abnormal_cov': abnormal_cov.cpu(),
            'abnormal_mean': abnormal_mean.cpu()
        }, save_path)
        
        print(f"Model saved to {save_path}")
        return save_path
    # ============ MAIN CODE STARTS HERE ============
    torch.autograd.set_detect_anomaly(True)
    config = args
    set_seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create dataset manager
    collate_fn = dataset.Collate(config)
    
    # Initialize model
    GNN = CensNet(config.input_dim, config.drop_out)
    transformer = TransformerBinaryClassifier(config, device, hidden_size=config.hidden_dim)
    transformer_d = PrototypeAttention()

    backbone = DGG(GNN, transformer, transformer_d)
    model = backbone.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Dictionary to store best validation metrics for each dataset
    best_val_aucs = {dataset_name: 0.0 for dataset_name in config.source_datasets}
    best_val_aps = {dataset_name: 0.0 for dataset_name in config.source_datasets}
    
    # Initialize prototype buffer - stores (normal_proto, abnormal_proto, difference_score)
    prototype_buffer = []
    difference_score = []
    buffer_size = 0  # Will be set based on dataset size
    current_dataset_idx = 0
    use_buffer = False  # Flag to use buffer after first few batches
    
    # For storing prototype relevance scores for datasets after the first one
    prototype_relevance_scores = []
    relevance_threshold = 0.5  # Threshold for prototype relevance
    best_prototype_idx = 0  # Index of the best prototype in the buffer
    
    # Keep track of training history
    trained_datasets = []
    
    # ====== SEQUENTIAL TRAINING ON SOURCE DATASETS ======
    print("Starting sequential training across source datasets...")
    config.data_set = config.target_datasets
    dataset_test = dataset.DygDataset(config, 'test')
    loader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_data_workers,
    collate_fn=collate_fn.dyg_collate_fn
    )
    
    for dataset_idx, dataset_name in enumerate(config.source_datasets):

        normal_mean = torch.zeros(config.input_dim).to('cuda')
        abnormal_mean = torch.zeros(config.input_dim).to('cuda')
        normal_cov = torch.eye(config.input_dim).to('cuda')
        abnormal_cov = torch.eye(config.input_dim).to('cuda')
        print(f"\n===== TRAINING ON SOURCE DATASET: {dataset_name} ({dataset_idx+1}/{len(config.source_datasets)}) =====")
        current_dataset_idx = dataset_idx
        

        
        # Create datasets
        config.data_set = dataset_name
        dataset_train = dataset.DygDataset(config, 'train')
        train_loader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_data_workers,
        pin_memory=True,
        collate_fn=collate_fn.dyg_collate_fn
    )
        # Set buffer size as 10% of training samples
        buffer_size = max(1, int(config.buffer_size * len(dataset_train)))
        #buffer_size = config.buffer_size
        print(f"Prototype buffer size set to {buffer_size}")
        
        # Create data loaders


        # For datasets after the first one, pre-calculate dataset embeddings and prototype relevance
        if dataset_idx > 0 and len(prototype_buffer) > 0:
            print(f"Pre-calculating dataset embeddings and prototype relevance scores...")
            dataset_embeddings = calculate_dataset_embeddings(train_loader, model, device, prototype_buffer)
            prototype_relevance_scores = calculate_prototype_relevance_scores(dataset_embeddings, prototype_buffer, device)
            
            # Find the best prototype based on combined score
            best_prototype_idx = 0
            difference_score = [proto[2] for proto in prototype_buffer]
            combined_scores = [config.relevance * rel + config.difference   * proto[2] for rel, proto in zip(prototype_relevance_scores, prototype_buffer)]
            print(f"Best prototype index: {best_prototype_idx}")
        
        # Initialize early stopping for this dataset
        early_stopper = EarlyStopMonitor(higher_better=True)
        best_dataset_val_auc = 0.0
        best_dataset_val_ap = 0.0
        patience = 5  # Number of epochs to wait for improvement
        counter = 0   # Counts epochs with no improvement
        best_loss = float('inf')
        # Training loop for current dataset
        for epoch in range(config.n_epochs):
            # Training phase
            model.train()
            total_loss = 0
            use_buffer = epoch > 0 or dataset_idx > 0  # Use buffer after first epoch or first dataset
            
            with tqdm(total=len(train_loader)) as t:
                for batch_idx, batch_sample in enumerate(train_loader):
                    t.set_description(f'Dataset {dataset_name} - Epoch {epoch}')

                    optimizer.zero_grad()
                    y, logits, output, normal_prompt, abnormal_prompt,nn, nc, ac,an = process_batch(batch_sample, model, device, normal_mean, normal_cov, abnormal_cov,abnormal_mean)
                    normal_mean = nn
                    abnormal_mean = an
                    normal_cov = nc
                    abnormal_cov = ac
                    # Extract normal and abnormal prototypes from this batch
                    normal_indices = torch.where(y==1)[0].tolist()
                    abnormal_indices = torch.where(y==0)[0].tolist()
                    
                    if len(normal_indices) > 0 and len(abnormal_indices) > 0:
                        normal_proto = output[normal_indices, :]
                        abnormal_proto = output[abnormal_indices, :]
                        
                        # Calculate differences between normal and abnormal prototypes
                        dif_normal = torch.sqrt(torch.sum((normal_prompt - normal_proto) ** 2, dim=1))
                        dif_abnormal = torch.sqrt(torch.sum((abnormal_prompt - abnormal_proto) ** 2, dim=1))
                        
                        # Calculate prototype difference score (mean Euclidean distance)
                        proto_diff_raw = torch.mean(torch.norm(normal_prompt - abnormal_prompt))
                        
                        # Normalize difference score to [0,1] range using sigmoid
                        proto_diff = torch.sigmoid(0.1 * proto_diff_raw).item()
                        
                        # Update prototype buffer
                        if len(prototype_buffer) < buffer_size:
                            # Buffer not full, add new pair
                            prototype_buffer.append((
                                normal_prompt.detach().cpu(), 
                                abnormal_prompt.detach().cpu(),
                                proto_diff
                            ))
                            # For second dataset onwards, assign relevance of 1.0 to new prototypes
                            
                            if dataset_idx > 0:
                                prototype_relevance_scores.append(1.0)
                                combined_scores.append(config.relevance * 1 + config.difference * proto_diff)
                                best_prototype_idx = np.argmax(combined_scores)
                            else:
                                difference_score.append(proto_diff)
                                best_prototype_idx = np.argmax(difference_score)
                            
                            
                        else:
                            # Buffer full, replace least different pair if this one is more different
                            # For non-first dataset, consider relevance too
                            if dataset_idx > 0:
                                worst_idx = np.argmin(combined_scores)
                                combined_scores_new = config.relevance * 1 + config.difference * proto_diff
                                if combined_scores_new>combined_scores[worst_idx]:
                                    prototype_buffer[worst_idx] = (
                                        normal_proto.mean(0).detach().cpu(),
                                        abnormal_proto.mean(0).detach().cpu(),
                                        proto_diff
                                    )
                                    prototype_relevance_scores[worst_idx] = 1.0  # New prototype has max relevance
                                    combined_scores[worst_idx] = combined_scores_new
                                    best_prototype_idx = np.argmax(combined_scores)
                            else:
                                # For first dataset, only consider difference
                                min_diff_idx = min(range(len(prototype_buffer)), 
                                                  key=lambda i: prototype_buffer[i][2])
                                if proto_diff > prototype_buffer[min_diff_idx][2]:
                                    prototype_buffer[min_diff_idx] = (
                                        normal_prompt.detach().cpu(),
                                        abnormal_prompt.detach().cpu(),
                                        proto_diff
                                    )
                                difference_score[min_diff_idx] = proto_diff
                                best_prototype_idx = np.argmax(difference_score)
                        
                        # Sort buffer by difference score (descending) for the first dataset
                        # For subsequent datasets, we keep the order as is since we're using relevance scores
                        if dataset_idx == 0:
                            prototype_buffer.sort(key=lambda x: x[2], reverse=True)
                        
                        loss_alignment = 1*torch.mean(dif_abnormal) + 0.1*torch.mean(dif_normal)
                    else:
                        # Handle case where batch doesn't have both classes
                        loss_alignment = torch.tensor(0.0).to(device)
                    
                    loss = args.ratio*criterion(logits, y) + (1-args.ratio)*loss_alignment
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    t.set_postfix(loss=loss.item())
                    t.update(1)
            
            avg_loss = total_loss / len(train_loader)
            print(f'Dataset {dataset_name} - Epoch {epoch} - Average training loss: {avg_loss:.4f}')
            
            # Print buffer info
            if dataset_idx == 0:
                print(f'Buffer size: {len(prototype_buffer)}, Top difference: {prototype_buffer[0][2] if prototype_buffer else "N/A"}')
            else:
                print(f'Buffer size: {len(prototype_buffer)}')
            if loss < best_loss:
                best_loss = loss
                counter = 0  # Reset counter if loss improves
            else:
                best_loss = loss
                counter += 1
                print(f'No improvement in loss for {counter} epoch(s).')
                if counter >= 4:
                    print(f'Early stopping: Validation loss did not improve for 4 consecutive epochs.')
                    break


        # Add current dataset to the list of trained datasets
        trained_datasets.append(dataset_name)
        
        # Save model after completing training on this dataset
        # This model has been trained on all datasets processed so far
        if trained_datasets:
            combined_dataset_name = "_".join(trained_datasets)

    # In your main loop, when you call save_model:
    save_model(model, optimizer, dataset_name, prototype_buffer, prototype_relevance_scores,
           normal_mean, normal_cov, abnormal_cov, abnormal_mean)
    test_loss, test_auc, test_ap = eval_model(loader_test,model, config, device, normal_mean, normal_cov, abnormal_cov,abnormal_mean)
    print('best auc:{}'.format(test_auc))
    print('best ap:{}'.format(test_ap))
    results_dir = "/home/jz/AAAI2025DGG-main/results"
    os.makedirs(results_dir, exist_ok=True)
    csv_filename = 'results_pretrain.csv'
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = [ 'auc', 'ap']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Write header only if file is new
    
    # Write results for each dataset

        writer.writerow({
            'auc': test_auc,
            'ap': test_ap

        })
            
        


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()