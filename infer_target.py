import sys
import torch
import torch.nn.functional as F
import datasets as dataset
import torch.utils.data
import sklearn
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import csv
import os
import pickle

from model.CensNet import CensNet
from model.DGG import DGG
from model.LSTM import LSTMBinaryClassifier
from option_infer_target import args
from utils import EarlyStopMonitor, logger_config
from tqdm import tqdm
import datetime, os
from model.Transformer import TransformerBinaryClassifier, PrototypeAttention
import random
from collections import defaultdict

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
        if len(prototype_buffer) > 0:
            # Use the best prototype
            best_pair_idx = best_prototype_idx
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
        m_loss, m_pred, m_label= np.array([]), np.array([]), np.array([])
        m_output = np.empty((128, 128))
        with torch.no_grad():
            model.eval()
            for batch_sample in data_loader:
                y, logits, output, _, _ ,_,_,_,_= process_batch(batch_sample, model, device, normal_mean, normal_cov, abnormal_cov,abnormal_mean)
                c_loss = np.array([criterion(logits, y).cpu()])
                
                pred_score = logits.cpu().numpy().flatten()
                y_np = y.cpu().numpy().flatten()

                m_loss = np.concatenate((m_loss, c_loss))
                m_pred = np.concatenate((m_pred, pred_score))
                m_label = np.concatenate((m_label, y_np))
                m_output = np.concatenate((m_output, output.cpu().numpy()), axis=0)

            no_ind = np.where(m_label == 0)[0].tolist()
            ano_ind = np.where(m_label == 1)[0].tolist()
            ind = no_ind+ano_ind
            oneind = no_ind+ano_ind[:int(len(no_ind)*0.005)]
            fiveind = no_ind+ano_ind[:int(len(no_ind)*0.007)]
            auc_roc = roc_auc_score(m_label, m_pred)
            avg_precision = average_precision_score(m_label, m_pred)
            aucone = roc_auc_score(m_label[oneind], m_pred[oneind])
            apone = average_precision_score(m_label[oneind], m_pred[oneind])
            aucfive = roc_auc_score(m_label[fiveind], m_pred[fiveind])
            apfive = average_precision_score(m_label[fiveind], m_pred[fiveind])


        return np.mean(m_loss), auc_roc, avg_precision, aucone, apone, aucfive, apfive, m_output, m_label

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

    def calculate_class_ranking_reliability(probs):
        """
        Calculates class-ranking reliability score for edges
        Lower entropy indicates higher reliability
        """
        # Calculate entropy: -p*log(p) - (1-p)*log(1-p)
        entropy = -(probs * torch.log(probs + 1e-10) + 
                   (1 - probs) * torch.log(1 - probs + 1e-10))
        return entropy

    def save_model(model, optimizer, val_metrics, dataset_name, prototype_buffer, prototype_relevance_scores, save_dir="/home/jz/AAAI2025DGG-main/save_model"):
        """Save model and related information"""
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{dataset_name}_best_model.pt")
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': val_metrics,
            'prototype_buffer': prototype_buffer,
            'prototype_relevance_scores': prototype_relevance_scores,
            'dataset_name': dataset_name
        }, save_path)
        
        print(f"Model saved to {save_path}")
        return save_path

    # ============ MAIN CODE STARTS HERE ============
    # Check if pretrained model path is provided
    pretrained_model_path = '/home/jz/AAAI2025DGG-main/save_model/Best_DNc.pt'
    
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
    
    # Initialize optimizer with empty parameter list (will be updated after loading checkpoint)
    optimizer = torch.optim.Adam([], lr=config.learning_rate)
    
    # Load pretrained model
    # Modified loading code
    checkpoint = torch.load(pretrained_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Load prototype buffer from checkpoint
    prototype_buffer = checkpoint.get('prototype_buffer', [])
    prototype_relevance_scores = checkpoint.get('prototype_relevance_scores', [])
    
    # Set buffer size as percentage of dataset
    buffer_size = max(1, config.buffer_size*len(prototype_buffer))
    
    best_prototype_idx = 0  # Will be updated
    relevance_threshold = 1
    
    # ============================================
    # PART 1: TRAINING ON TARGET DATASET (NO LABELS)
    # ============================================
    print("\n===== TRAINING ON TARGET DATASET (NO LABELS) =====")
    
    target_trained_datasets = []
    best_model_paths = {}  # Store paths to best models for each target dataset
    
    
    normal_mean = checkpoint.get('normal_mean', torch.zeros(args.input_dim)).to(device)
    normal_cov = checkpoint.get('normal_cov', torch.eye(args.input_dim)).to(device)
    abnormal_cov = checkpoint.get('abnormal_cov', torch.eye(args.input_dim)).to(device)
    abnormal_mean = checkpoint.get('abnormal_mean', torch.zeros(args.input_dim)).to(device)
    
    # Make prototype statistics trainable
    normal_mean.requires_grad = True
    normal_cov.requires_grad = True
    abnormal_mean.requires_grad = True
    abnormal_cov.requires_grad = True
    
    # Update optimizer to only optimize prototype statistics
    optimizer = torch.optim.Adam([normal_mean, normal_cov, abnormal_mean, abnormal_cov], lr=config.learning_rate)
    
    print(f"Target dataset: {config.target_datasets}")
    print("Model parameters frozen, only prototype statistics will be updated during training")

    
    # Get all indices for target dataset

    
    # Create target dataset
    config.data_set = config.target_datasets
    dataset_test = dataset.DygDataset(config, 'train')
    target_loader = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_data_workers,
    collate_fn=collate_fn.dyg_collate_fn
)
    
    # Pre-calculate embeddings and find best prototype for target dataset
    if len(prototype_buffer) > 0:
        print("Calculating initial embeddings for target dataset...")
        # Since this is an unsupervised setting, we'll use a different approach
        # We'll run a forward pass on the entire dataset and get embeddings
        
        all_embeddings = []
        model.eval()
        with torch.no_grad():
            # Use a fixed prototype for consistent embeddings
            normal_prompt_raw = prototype_buffer[0][0].to(device)
            abnormal_prompt_raw = prototype_buffer[0][1].to(device)
            
            for batch_sample in tqdm(target_loader, desc="Calculating target embeddings"):
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
                    abnormal_prompt_raw
                    ,
                normal_mean, normal_cov, abnormal_cov,abnormal_mean
                )
                
                all_embeddings.append(output.cpu())
        
        # Concatenate all embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0) if all_embeddings else torch.tensor([])
        
        # Calculate relevance scores for each prototype
        # Since we don't have labels, we'll compute general closeness
        relevance_scores = []
        
        if len(all_embeddings) > 0:
            all_embeddings = all_embeddings.to(device)
            
            for n_proto, a_proto, diff_score in prototype_buffer:
                n_proto = n_proto.to(device)
                a_proto = a_proto.to(device)
                
                # Calculate average distance to both prototypes
                n_distances = torch.norm(all_embeddings - n_proto.unsqueeze(0), dim=1)
                a_distances = torch.norm(all_embeddings - a_proto.unsqueeze(0), dim=1)
                
                # Use minimum distance to either prototype
                min_distances, _ = torch.min(torch.stack([n_distances, a_distances]), dim=0)
                relevance = -torch.mean(min_distances).item()  # Negative distance = higher relevance
                
                relevance_scores.append(relevance)
            
            # Normalize to [0,1]
            min_score = min(relevance_scores)
            max_score = max(relevance_scores)
            
            if max_score > min_score:
                relevance_scores = [config.no_rel*(score - min_score) / (max_score - min_score) for score in relevance_scores]
            else:
                relevance_scores = [0.5] * len(relevance_scores)
            
            # Combine with difference scores
            combined_scores = [config.relevance * rel + config.difference * proto[2] for rel, proto in zip(relevance_scores, prototype_buffer)]
            best_prototype_idx = np.argmax(combined_scores)
            
            #print(f"Target dataset prototype relevance scores: {relevance_scores}")
            print(f"Best prototype for target dataset: {best_prototype_idx}")
    
    # Train model using online pseudo-labeling and prototype alignment
    best_target_loss = float('inf')
    best_model_path = None
    show_confident = []
    for epoch in range(config.n_epochs):
        total_loss = 0
        model.train()
        
        with tqdm(total=len(target_loader)) as t:
            for batch_idx, batch_sample in enumerate(target_loader):
                t.set_description(f'Target Dataset {config.target_datasets} - Epoch {epoch}')
                
                # Process batch
                optimizer.zero_grad()
                y, logits, output, normal_prompt, abnormal_prompt,nn, nc, ac,an = process_batch(batch_sample, model, device, normal_mean, normal_cov, abnormal_cov,abnormal_mean)
                normal_mean = nn
                abnormal_mean = an
                normal_cov = nc
                abnormal_cov = ac
                # Create pseudo-labels based on prediction confidence
                edge_probs = torch.sigmoid(logits)
                edge_entropy = calculate_class_ranking_reliability(edge_probs)
                
                # Sort edges within batch by reliability (lower entropy = more reliable)
                sorted_entropy, sorted_indices = torch.sort(edge_entropy)
                
                # Take top 10% as normal and bottom 10% as abnormal
                batch_size = edge_entropy.size(0)
                normal_count = max(1, int(batch_size * config.confident))
                abnormal_count = max(1, int(batch_size * config.confident))
                
                # Create pseudo-labels
                normal_indices = sorted_indices[:normal_count]
                abnormal_indices = sorted_indices[-abnormal_count:]
                show_confident.append({'fn':y[normal_indices].cpu(),'gn':torch.ones(12),'nr':output[normal_indices].cpu(),'fa':y[abnormal_indices].cpu(),'ga':torch.zeros(12),'ar':output[abnormal_indices].cpu()})
                # Use pseudo-labels only for prototype alignment
                if len(normal_indices) > 0 and len(abnormal_indices) > 0:
                    normal_proto = output[normal_indices]
                    abnormal_proto = output[abnormal_indices]
                    
                    # Calculate alignment loss
                    dif_normal = torch.sqrt(torch.sum((normal_prompt - normal_proto) ** 2, dim=1))
                    dif_abnormal = torch.sqrt(torch.sum((abnormal_prompt - abnormal_proto) ** 2, dim=1))
                    
                    loss_alignment = torch.mean(dif_abnormal) + 0.1*torch.mean(dif_normal)
                    
                    # Update prototype buffer
                    proto_diff_raw = torch.mean(torch.norm(normal_proto.mean(0) - abnormal_proto.mean(0)))
                    proto_diff = torch.sigmoid(0.1 * proto_diff_raw).item()
                    
                    if len(prototype_buffer) < buffer_size:
                        prototype_buffer.append((
                            normal_proto.mean(0).detach().cpu(),
                            abnormal_proto.mean(0).detach().cpu(),
                            proto_diff
                        ))
                        relevance_scores.append(1.0)  # New prototype from target has max relevance
                        combined_scores.append(config.relevance * 1 + config.difference * proto_diff)
                        best_prototype_idx = np.argmax(combined_scores)
                    else:
                        # Find prototype with lowest combined score
                        worst_idx = np.argmin(combined_scores)
                        combined_scores_new = config.relevance * 1 + config.difference * proto_diff
                        if combined_scores_new>combined_scores[worst_idx]:
                            prototype_buffer[worst_idx] = (
                                normal_proto.mean(0).detach().cpu(),
                                abnormal_proto.mean(0).detach().cpu(),
                                proto_diff
                            )
                            relevance_scores[worst_idx] = 1.0  # New prototype has max relevance
                            combined_scores[worst_idx] = combined_scores_new
                            best_prototype_idx = np.argmax(combined_scores)
                    
                    # Use only alignment loss (no logit loss with pseudo-labels)
                    loss = loss_alignment
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    t.set_postfix(loss=loss.item())
                else:
                    t.set_postfix(loss=0.0)
                    
                t.update(1)
        
        avg_loss = total_loss / len(target_loader) if len(target_loader) > 0 else 0
        print(f'Target Dataset {config.target_datasets} - Epoch {epoch} - Average training loss: {avg_loss:.4f}')
        print(f'Buffer size: {len(prototype_buffer)}, Best prototype: {best_prototype_idx}')
        
        # Save model if this is the best loss on target dataset

            
    # Store the path to the best model for this dataset

    torch.save(show_confident,'/home/jz/AAAI2025DGG-main/cv_dnc.pt')
    
    print("\nTraining on target datasets completed!")
    
    # ============================================
    # PART 2: TESTING ON TARGET DATASET
    # ============================================
    print("\n===== TESTING ON TARGET DATASET =====")
    
    # Find best prototype based on relevance scores for testing
    if prototype_buffer and prototype_relevance_scores:
        best_prototype_idx = np.argmax(combined_scores)
        print(f"Using best prototype index {best_prototype_idx} for testing")
    else:
        best_prototype_idx = 0
        print("No prototype buffer available, using default prototype")
    
    test_results = {}

    print(f"\nEvaluating on {config.target_datasets} test set:")
    
    # Get test indices

    
    # Create test dataset and loader
    config.data_set = config.target_datasets
    dataset_test = dataset.DygDataset(config, 'test')
    test_loader = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_data_workers,
    collate_fn=collate_fn.dyg_collate_fn
)
    
    # Evaluate on test set
    test_loss, test_auc, test_ap, aucone, apone, aucfive, apfive, m_output, m_label = eval_model(test_loader, model, config, device,normal_mean, normal_cov, abnormal_cov,abnormal_mean)
    np.save('/home/jz/AAAI2025DGG-main/representation/dnc_representation.npy', m_output)
    np.save('/home/jz/AAAI2025DGG-main/representation/dnc_label.npy', m_label)

    test_results[config.target_datasets] = {
        'loss': test_loss,
        'auc': test_auc,
        'ap': test_ap,
        'aucone':aucone,
        'apone':apone,
        'aucfive':aucfive,
        'apfive':apfive
    }
    print(f' Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}, Test AUCone: {aucone:.4f}, Test APone: {apone:.4f}, Test AUCfive: {aucfive:.4f}, Test APfive: {apfive:.4f}')
    
    # Print summary of results
    print("\n===== SUMMARY OF TEST RESULTS =====")
    for dataset_name, metrics in test_results.items():
        print(f"{dataset_name}: AUC = {metrics['auc']:.4f}, AP = {metrics['ap']:.4f}, Loss = {metrics['loss']:.4f}")
    results_dir = "/home/jz/AAAI2025DGG-main/results"
    os.makedirs(results_dir, exist_ok=True)
    csv_filename = 'results_dnc.csv'
    with open('/home/jz/AAAI2025DGG-main/prototype.pkl','wb') as file:
        pickle.dump(prototype_buffer,file)
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = [ 'auc', 'ap', 'loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header only if file is new
        
        # Write results for each dataset
        for dataset_name, metrics in test_results.items():
            writer.writerow({
                'auc': f"{metrics['auc']:.4f}",
                'ap': f"{metrics['ap']:.4f}"
            })
            
        # Save final model with test results


if __name__ == "__main__":
    main()
