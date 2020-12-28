import torch

import random
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_induced', type=bool, default='True')
    parser.add_argument('--source_file', choices=['mimic', 'eicu', 'both'], type=str, default='eicu')
    parser.add_argument('--target', choices=['readmission', 'mortality', 'los>3day', 'los>7day', 'diagnosis'], type=str, default='readmission')
    parser.add_argument('--item', choices=['lab', 'diagnosis', 'chartevent', 'medication', 'infusion'], type=str, default='lab')
    parser.add_argument('--time_window', choices=['12', '24', '36', '48', 'Total'], type=str, default='12')
    parser.add_argument('--rnn_model_type', choices=['gru', 'lstm'], type=str, default='gru')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--rnn_bidirection', type=bool, default=True)
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_length', type=str, default='150')
    parser.add_argument('--bert_model', type=str, default='clinical_bert')
    parser.add_argument('--bert_freeze', type=bool, default=True)
    parser.add_argument('--path', type=str, default='./')
    parser.add_argument('--filename', type=str, default='tester')
    args = parser.parse_args()

    torch.cuda.empty_cache()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.bert_induced:
        from dataset.prebert_dataloader import bertinduced_get_dataloader as get_dataloader
        from trainer.bert_induced_trainer import Bert_Trainer as Trainer
    else:
        from dataset.singlernn_dataloader import singlernn_get_dataloader as get_dataloader
        from trainer.base_trainer import Trainer



    if args.time_window == '12':
        assert args.max_length == '150', "time_window of 12 should have max length of 150!"
    elif args.time_window == '24':
        assert args.max_length == '200', "time_window of 24 should have max length of 200!"


    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    for valid_index in range(5):
        valid_index += 1
        train_loader = get_dataloader(args=args, validation_index=valid_index, data_type='train')
        valid_loader = get_dataloader(args=args, validation_index=valid_index, data_type='eval')

        trainer = Trainer(args, train_loader, valid_loader, device)
        trainer.train()


if __name__ == '__main__':
    main()