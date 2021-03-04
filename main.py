import torch
import torch.multiprocessing as mp

import random
import numpy as np
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--DescEmb', action='store_true', help='True if DescEmb, False if CodeEmb')
    parser.add_argument('--source_file', choices=['mimic', 'eicu', 'both'], type=str, default='mimic', help='both for pooling')
    parser.add_argument('--target', choices=['readmission', 'mortality', 'los>3day', 'los>7day', 'dx_depth1_unique'], type=str, default='readmission')
    parser.add_argument('--item', choices=['all'], type=str, default='lab')
    parser.add_argument('--time_window', choices=['12', '24', '36', '48', 'Total'], type=str, default='12')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_length', type=str, default='150')
    parser.add_argument('--bert_model', choices=['bert', 'bio_clinical_bert', 'bio_bert', 'pubmed_bert', 'blue_bert', 'bert_mini', 'bert_tiny', 'bert_small'], type=str)
    parser.add_argument('--cls_freeze', action='store_true')
    parser.add_argument('--input_path', type=str, default='/home/jylee/data/pretrained_ehr/input_data/', help='data directory')
    parser.add_argument('--path', type=str, default='/home/jylee/data/pretrained_ehr/output/KDD_output/', help='model saving directory')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.DescEmb:
        from dataset.DescEmb_dataloader import DescEmb_get_dataloader as get_dataloader
        from trainer.DescEmb_trainer import DescEmb_Trainer as Trainer
        if args.cls_freeze:
            print('DesEmb-FR')
        else:
            print('DescEmb-FT')

    elif not args.DescEmb:
        from dataset.CodeEmb_dataloader import CodeEmb_get_dataloader as get_dataloader
        from trainer.CodeEmb_trainer import Trainer
        print('CodeEmb')


    mp.set_sharing_strategy('file_system')

    SEED = [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029]

    for seed in SEED:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True

        args.seed = seed
        print('seed_number', args.seed)

        train_loader = get_dataloader(args=args, data_type='train')

        trainer = Trainer(args, train_loader, device)
        trainer.train()

        print('Finished training seed: {}'.format(seed))

if __name__ == '__main__':
    main()
