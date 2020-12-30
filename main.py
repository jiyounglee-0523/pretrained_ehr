import torch
from test import Tester
import random
import numpy as np
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'],type=str, default='train')
    parser.add_argument('--bert_induced', action='store_true')
    parser.add_argument('--source_file', choices=['mimic', 'eicu', 'both'], type=str, default='eicu')
    parser.add_argument('--target', choices=['readmission', 'mortality', 'los>3day', 'los>7day', 'dx_depth1_unique'], type=str, default='dx_depth1_unique')
    parser.add_argument('--item', choices=['lab', 'diagnosis', 'chartevent', 'medication', 'infusion'], type=str, default='lab')
    parser.add_argument('--time_window', choices=['12', '24', '36', '48', 'Total'], type=str, default='12')
    parser.add_argument('--rnn_model_type', choices=['gru', 'lstm'], type=str, default='gru')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--rnn_bidirection', action='store_true')
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_length', type=str, default='150')
    parser.add_argument('--bert_model', type=str, default='clinical_bert')
    parser.add_argument('--bert_freeze', action='store_true')
    parser.add_argument('--path', type=str, default='/home/jylee/data/pretrained_ehr/output/arxiv_output/')
    parser.add_argument('--word_max_length', type=int, default=15)    # tokenized word max_length, used in padding
    args = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    if args.bert_induced:
        from dataset.prebert_dict_dataloader import bertinduced_dict_get_dataloader as get_dataloader
        from trainer.bert_dict_trainer import bert_dict_Trainer as Trainer
        print('bert induced')
    else:
        from dataset.singlernn_dataloader import singlernn_get_dataloader as get_dataloader
        from trainer.base_trainer import Trainer
        print('single_rnn')



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

    if args.mode == 'train':
        for valid_index in range(5):
            valid_index += 1
            train_loader = get_dataloader(args=args, validation_index=valid_index, data_type='train')
            valid_loader = get_dataloader(args=args, validation_index=valid_index, data_type='eval')

            trainer = Trainer(args, train_loader, valid_loader, device, valid_index)
            trainer.train()

            print('Finished training valid_index: {}'.format(valid_index))

    elif args.mode == 'test':
        print('Test start_{}_{}_{}_bert_induced_{}_dropout{}_emb{}_hid{}_bidirect{}_lr{}'.format(
            args.source_file, args.item, args.target, args.bert_induced,args.dropout, args.embedding_dim, args.hidden_dim, args.rnn_bidirection, args.lr))

        test_loader = get_dataloader(args=args, validation_index=0, data_type='test')
        tester = Tester(args, test_loader, device)
        tester.test()

        print('Finished test!')

if __name__ == '__main__':
    main()