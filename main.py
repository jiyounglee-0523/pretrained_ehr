import torch
import torch.multiprocessing as mp

import random
import numpy as np
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_induced', action='store_true')
    parser.add_argument('--source_file', choices=['mimic', 'eicu', 'both'], type=str, default='mimic')
    parser.add_argument('--target', choices=['readmission', 'mortality', 'los>3day', 'los>7day', 'dx_depth1_unique'], type=str, default='readmission')
    parser.add_argument('--item', choices=['lab', 'med', 'inf'], type=str, default='lab')
    parser.add_argument('--time_window', choices=['12', '24', '36', '48', 'Total'], type=str, default='12')
    parser.add_argument('--rnn_model_type', choices=['gru', 'lstm'], type=str, default='gru')
    parser.add_argument('--batch_size', type=int, default=256)
    # parser.add_argument('--dropout', type=float, default=0.1)
    # parser.add_argument('--embedding_dim', type=int, default=768)
    # parser.add_argument('--hidden_dim', type=int, default=512)
    # parser.add_argument('--rnn_bidirection', action='store_true')
    parser.add_argument('--n_epochs', type=int, default=500)
    # parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max_length', type=str, default='150')
    parser.add_argument('--bert_model', choices=['bio_clinical_bert', 'bio_bert', 'pubmed_bert', 'blue_bert'], type=str)
    parser.add_argument('--bert_freeze', action='store_true')
    parser.add_argument('--path', type=str, default='/home/jylee/data/pretrained_ehr/output/KDD_output/')
    parser.add_argument('--word_max_length', type=int, default=15)    # tokenized word max_length, used in padding
    parser.add_argument('--device_number', type=int)
    args = parser.parse_args()

    # args.device_number = 6
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_number)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.rnn_bidirection = False

    # hyperparameter tuning
    args.dropout = 0.3
    args.embedding_dim = 128
    args.hidden_dim = 256
    args.lr = 1e-4


    if args.bert_induced and args.bert_freeze:
        from dataset.prebert_dict_dataloader import bertinduced_dict_get_dataloader as get_dataloader
        from trainer.bert_dict_trainer import bert_dict_Trainer as Trainer
        print('bert induced')

    elif args.bert_induced and not args.bert_freeze:
        from dataset.prebert_dataloader import bertinduced_get_dataloader as get_dataloader
        from trainer.bert_induced_trainer import Bert_Trainer as Trainer
        print('bert finetune')

    elif not args.bert_induced:
        from dataset.singlernn_dataloader import singlernn_get_dataloader as get_dataloader
        from trainer.base_trainer import Trainer
        print('single_rnn')

    if args.time_window == '12':
        assert args.max_length == '150', "time_window of 12 should have max length of 150!"
    elif args.time_window == '24':
        assert args.max_length == '200', "time_window of 24 should have max length of 200!"

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

        # change the train_loader, valid_loader (Dataset)   valid_index should be changed!
        train_loader = get_dataloader(args=args, data_type='train')
        valid_loader = get_dataloader(args=args, data_type='eval')

        trainer = Trainer(args, train_loader, valid_loader, device)
        trainer.train()

        print('Finished training seed: {}'.format(seed))

if __name__ == '__main__':
    main()