import argparse

parser = argparse.ArgumentParser(description='Denoise')
parser.add_argument('--dir_data', type=str, default='./data')
parser.add_argument('--name_pos', type=str, default='EU3')
parser.add_argument('--ratio_neg', type=str, default='1')
parser.add_argument('--mode', type=str, default='Target')

parser.add_argument('--source_datasets', type=str, nargs='+', default=['MOOC','Wikipedia'],
                    help='List of datasets to train and test on')
parser.add_argument('--target_datasets', type=str, nargs='+', default='dnc_10percent', 
                    help='List of datasets to train and test on')



parser.add_argument('--data_set', type=str, default='dnc_10percent')
parser.add_argument('--neg', type=str, default='01') 
parser.add_argument('--max_len', type=int, default=24)  
parser.add_argument('--ratio', type=float, default=0.1)  
parser.add_argument('--no_rel', type=float, default=0.5)
parser.add_argument('--relevance', type=float, default=0.7)
parser.add_argument('--difference', type=float, default=0.3)  

##data param
parser.add_argument('--batch_size', type=int, default=512) 
parser.add_argument('--n_epochs', type=int, default=50)
parser.add_argument('--num_data_workers', type=int, default=0)  
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--buffer_size', type=int, default=0.1)
##model param
parser.add_argument('--ckpt_file', type=str, default='./')
parser.add_argument('--input_dim', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=258)  
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--drop_out', type=float, default=0.4)
parser.add_argument('--csratio', type=float, default=0.4)
parser.add_argument('--n_layer', type=int, default=6, help='Number of network layers')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--seed', type=int, default=95540)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--confident', type=float, default=0.1)  
parser.add_argument('--confident_detection_method', type=str, default='entropy',
                    choices=['entropy', 'random', 'threshold', 'distance', 'similarity'],
                    help='Method for confident detection selection in ablation study')

args = parser.parse_args()