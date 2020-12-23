import torch

import random
import numpy as np
import argparse
import wandb

from dataset.dataloader import get_dataloader
from trainer.base_trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_file', choices=['mimic', 'eicu', 'both'], type=str, default='mimic')
    parser.add_argument('--target', choices=['readmission', 'mortality', 'los>3day', 'los>7day', 'diagnosis'], type=str, default='readmission')
    parser.add_argument('--item', choices=['lab', 'diagnosis','chartevent','medication','infusion'], type=str, default='lab')
    parser.add_argument('--time_window', choices=['12', '24', '36', '48', 'Total'], type=str, default='24')
    parser.add_argument('--model_type', choices=['gru', 'lstm'], type=str, default='gru')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--rnn_bidirection', type=bool, default=True)
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_length', type=str, default='200')
    args = parser.parse_args()



    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    train_loader = get_dataloader(args, data_type='train')
    valid_loader = get_dataloader(args, data_type='eval')

    trainer = Trainer(args, train_loader, valid_loader, device)
    trainer.train()


if __name__ == '__main__':
    main()