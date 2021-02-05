import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.multiprocessing as mp

from sklearn.metrics import roc_auc_score, average_precision_score

import os
import argparse
import pickle
import random
import wandb
import tqdm
import numpy as np

from test import Few_Shot_Dataset
from utils.loss import *
from models.rnn_bert_dict import *
from models.rnn_models import *
from models.prebert import *
from models.transformer import *
from utils.trainer_utils import *


def get_dataloader(args, data_type, data_name=None):
    if data_type == 'train':
        train_data = Portion_Dataset(args, data_type=data_type)
        dataloader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=16)

    elif data_type == 'eval':
        eval_data = Portion_Dataset(args, data_type=data_type, data_name=data_name)
        dataloader = DataLoader(dataset=eval_data, batch_size=args.batch_size, shuffle=True, num_workers=16)

    elif data_type == 'test':
        test_data = Portion_Dataset(args, data_type=data_type, data_name=data_name)
        dataloader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, num_workers=16)
    return dataloader


class Portion_Dataset(Few_Shot_Dataset):
    def __init__(self, args, data_type, data_name=None):
        args.test_file = args.source_file
        super(Portion_Dataset, self).__init__(args=args, data_type=data_type, data_name=data_name)

################################################################################################################

class DataSize_Trainer():
    def __init__(self, args, train_dataloader, device):
        self.train_dataloader = train_dataloader
        if args.source_file != 'both':
            self.test_dataloader = get_dataloader(args, data_type='test')
            self.valid_dataloader = get_dataloader(args, data_type='eval')
        elif args.source_file == 'both':
            self.mimic_test_dataloader = get_dataloader(args, data_type='test', data_name='mimic')
            self.mimic_valid_dataloader = get_dataloader(args, data_type='eval', data_name='mimic')
            self.eicu_test_dataloader = get_dataloader(args, data_type='test', data_name='eicu')
            self.eicu_valid_dataloader = get_dataloader(args, data_type='eval', data_name='eicu')

        self.device = device
        self.debug = args.debug

        if not self.debug:
            wandb.init(project=args.wandb_project_name, entity='pretrained_ehr', config=args, reinit=True)

        lr = args.lr
        self.n_epochs = args.n_epochs
        self.BCE = args.only_BCE
        self.target = args.target
        self.source_file = args.source_file

        if args.only_BCE:
            self.criterion = nn.BCEWithLogitsLoss()
            if args.target == 'dx_depth1_unique':
                output_size = 18
            else:
                output_size = 1
        elif not args.only_BCE:
            if args.target == 'dx_depth1_unique':
                output_size = 18
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = FocalLoss()
                output_size = 18

        file_target_name = args.target
        if file_target_name == 'los>3day':
            file_target_name = 'los_3days'
        elif file_target_name == 'los>7day':
            file_target_name = 'los_7days'

        if args.bert_induced and args.bert_freeze and not args.cls_freeze and not args.transformer:
            model_directory = 'cls_learnable'
            self.model = dict_post_RNN(args=args, output_size=output_size, device=self.device, target_file=args.source_file).to(device)
            print('bert freeze, cls_learnable, RNN')
            if args.concat:
                filename = 'cls_learnable_rnn_{}_{}_dataportion{}_concat'.format(args.bert_model, args.seed, args.few_shot)
            elif not args.concat:
                filename = 'cls_learnable_rnn_{}_{}_dataportion{}'.format(args.bert_model, args.seed, args.few_shot)

        elif args.bert_induced and args.bert_freeze and not args.cls_freeze and args.transformer:
            model_directory = 'cls_learnable'
            self.model = Transformer(args=args, output_size=output_size, device=self.device, target_file=args.source_file).to(self.device)
            if not args.concat:
                filename = 'cls_learnable_transformer_layer{}_attnheads{}_hidden{}_{}_{}_dataportion{}_onlyBCE'.format(args.transformer_layers, args.transformer_attn_heads,
                                                                                                            args.transformer_hidden_dim, args.bert_model, args.seed, args.few_shot)

        elif args.bert_induced and args.bert_freeze and args.cls_freeze and not args.transformer:
            model_directory = 'cls_learnable'
            self.model = dict_post_RNN(args=args, output_size=output_size, device=self.device, target_file=args.source_file).to(device)
            print('bert freeze, cls freeze')
            if args.concat:
                filename = 'cls_fixed_rnn_{}_{}_dataportion{}_concat'.format(args.bert_model, args.seed, args.few_shot)
            elif not args.concat:
                filename = 'cls_fixed_rnn_{}_{}_dataportion{}'.format(args.bert_model, args.seed, args.few_shot)

        elif args.bert_induced and not args.bert_freeze and not args.transformer:
            model_directory = 'bert_finetune'
            self.model = post_RNN(args=args, output_size=output_size).to(device)
            print('bert_finetuning')
            ## model name!!

        elif not args.bert_induced:
            model_directory = 'singleRNN'
            if args.source_file == 'mimic':
                if args.item == 'lab':
                    vocab_size = 5110 if args.concat else 359
                elif args.item == 'med':
                    vocab_size = 2211 if args.concat else 1535
                elif args.item == 'inf':
                    vocab_size = 485
                elif args.item == 'all':
                    vocab_size = 7563 if args.concat else 2377
            elif args.source_file == 'eicu':
                if args.item == 'lab':
                    vocab_size = 9659 if args.concat else 134
                elif args.item == 'med':
                    vocab_size = 2693 if args.concat else 1283
                elif args.item == 'inf':
                    vocab_size = 495
                elif args.item == 'all':
                    vocab_size = 8532 if args.concat else 1344
            elif args.source_file == 'both':
                if args.item == 'lab':
                    vocab_size = 14371 if args.concat else 448
                elif args.item == 'med':
                    vocab_size = 4898 if args.concat else 2812
                elif args.item == 'inf':
                    vocab_size = 979
                elif args.item == 'all':
                    vocab_size = 15794 if args.concat else 3672

            if not args.transformer:
                self.model = RNNmodels(args, vocab_size, output_size, self.device).to(device)
                print('singleRNN')

                if args.concat:
                    filename = 'trained_single_rnn_{}_dataportion{}_concat'.format(args.seed, args.few_shot)
                elif not args.concat:
                    filename = 'trained_single_rnn_{}_dataportion{}'.format(args.seed, args.few_shot)
            elif args.transformer:
                self.model = Transformer(args, output_size, device, target_file=args.source_file, vocab_size=vocab_size).to(self.device)
                print('Transformer')
                if not args.concat:
                    filename = 'trained_transformer_layers{}_attnheads{}_hidden{}_{}_dataportion{}_onlyBCE'.format(args.transformer_layers, args.transformer_attn_heads,
                                                                                                                   args.transformer_hidden_dim, args.seed, args.few_shot)

        path = os.path.join(args.path, args.item, model_directory, args.source_file, file_target_name, filename)
        print('Model will be saved in {}'.format(path))

        self.best_eval_path = path + '_best_auprc.pt'
        self.best_mimic_eval_path = path + '_mimic_best_auprc.pt'
        self.best_eicu_eval_path = path + '_eicu_best_auprc.pt'
        self.final_path = path + '_final.pt'

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.early_stopping = EarlyStopping(patience=50, verbose=True)
        self.mimic_early_stopping = EarlyStopping(patience=50, verbose=True)
        self.eicu_early_stopping = EarlyStopping(patience=50, verbose=True)
        num_params = count_parameters(self.model)
        print('Number of parameters: {}'.format(num_params))

    def train(self):
        best_loss = float('inf')
        best_mimic_loss = float('inf')
        best_eicu_loss = float('inf')
        best_auroc = 0.0
        best_mimic_auroc = 0.0
        best_eicu_auroc = 0.0
        best_auprc = 0.0
        best_mimic_auprc = 0.0
        best_eicu_auprc = 0.0

        for n_epoch in range(self.n_epochs + 1):
            preds_train = []
            truths_train = []
            avg_train_loss = 0.

            for iter, sample in tqdm.tqdm(enumerate(self.train_dataloader)):
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                item_id, item_target, seq_len = sample
                item_id = item_id.to(self.device)
                item_target = item_target.to(self.device)

                y_pred = self.model(item_id, seq_len)

                if self.BCE and self.target != 'dx_depth1_unique':
                    loss = self.criterion(y_pred, item_target.unsqueeze(1).float().to(self.device))
                else:
                    loss = self.criterion(y_pred, item_target.float().to(self.device))

                loss.backward()
                self.optimizer.step()

                avg_train_loss += loss.item() / len(self.train_dataloader)

                probs_train = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_train += list(probs_train.flatten())
                truths_train += list(item_target.detach().cpu().numpy().flatten())

            auroc_train = roc_auc_score(truths_train, preds_train)
            auprc_train = average_precision_score(truths_train, preds_train, average='micro')

            if self.source_file != 'both':
                avg_eval_loss, auroc_eval, auprc_eval = self.evaluation()

                if best_auprc < auprc_eval:
                    best_loss = avg_eval_loss
                    best_auroc = auroc_eval
                    best_auprc = auprc_eval
                    if not self.debug:
                        torch.save({'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict(),
                                    'loss': best_loss,
                                    'auroc': best_auroc,
                                    'auprc': best_auprc,
                                    'epochs': n_epoch}, self.best_eval_path)
                        print('Model parameter saved at epoch {}'.format(n_epoch))

                if not self.debug:
                    wandb.log({'train_loss': avg_train_loss,
                               'train_auroc': auroc_train,
                               'train_auprc': auprc_train,
                               'eval_loss': avg_eval_loss,
                               'eval_auroc': auroc_eval,
                               'eval_auprc': auprc_eval})

                print('[Train]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_train_loss, auroc_train, auprc_train))
                print('[Valid]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_eval_loss, auroc_eval, auprc_eval))

                self.early_stopping(auprc_eval)
                if self.early_stopping.early_stop:
                    print('Early stopping')
                    if not self.debug:
                        torch.save({'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict(),
                                    'loss': avg_eval_loss,
                                    'auroc': best_auroc,
                                    'auprc': best_auprc,
                                    'epochs': n_epoch}, self.final_path)

                    self.test()
                    break

            elif self.source_file == 'both':
                mimic_avg_eval_loss, mimic_auroc_eval, mimic_auprc_eval, eicu_avg_eval_loss, eicu_auroc_eval, eicu_auprc_eval = self.evaluation_both()

                if best_mimic_auprc < mimic_auprc_eval:
                    best_mimic_loss = mimic_avg_eval_loss
                    best_mimic_auroc = mimic_auroc_eval
                    best_mimic_auprc = mimic_auprc_eval
                    if not self.debug:
                        torch.save({'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict(),
                                    'loss': best_mimic_loss,
                                    'auroc': best_mimic_auroc,
                                    'auprc': best_mimic_auprc,
                                    'epochs': n_epoch}, self.best_mimic_eval_path)
                    print('[mimic] Model parameter saved at epoch {}'.format(n_epoch))

                    if not self.debug:
                        wandb.log({'train_loss': avg_train_loss,
                                   'train_auroc': auroc_train,
                                   'train_auprc': auprc_train})

                if best_eicu_auprc < eicu_auprc_eval:
                    best_eicu_loss = eicu_avg_eval_loss
                    best_eicu_auroc = eicu_auroc_eval
                    best_eicu_auprc = eicu_auprc_eval
                    if not self.debug:
                        torch.save({'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict(),
                                    'loss': best_eicu_loss,
                                    'auroc': best_eicu_auroc,
                                    'auprc': best_eicu_auprc,
                                    'epochs': n_epoch}, self.best_eicu_eval_path)
                    print('[eicu] Model parameter saved at epoch {}'.format(n_epoch))

                print('[Train]  loss: {:.3f},  auroc: {:.3f},   auprc: {:.3f}'.format(avg_train_loss, auroc_train, auprc_train))
                print('[mimic/Valid]  loss: {:.3f},  auroc: {:.3f},   auprc: {:.3f}'.format(mimic_avg_eval_loss, mimic_auroc_eval, mimic_auprc_eval))
                print('[eicu/Valid]  loss: {:.3f},  auroc: {:.3f},   auprc: {:.3f}'.format(eicu_avg_eval_loss, eicu_auroc_eval, eicu_auprc_eval))

                self.mimic_early_stopping(mimic_auprc_eval)
                self.eicu_early_stopping(eicu_auprc_eval)

                if self.mimic_early_stopping.early_stop and self.eicu_early_stopping.early_stop:
                    print('Early stopping')

                    if not self.debug:
                        torch.save({'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict(),
                                    'epochs': n_epoch}, self.final_path)
                    break

        if self.source_file != 'both':
            self.test()
        elif self.source_file == 'both':
            self.test_both()


    def evaluation(self):
        self.model.eval()

        preds_eval = []
        truths_eval = []
        avg_eval_loss = 0.

        with torch.no_grad():
            for iter, sample in enumerate(self.valid_dataloader):
                item_id, item_target, seq_len = sample
                item_id = item_id.to(self.device)
                item_target = item_target.to(self.device)

                y_pred = self.model(item_id, seq_len)
                if self.BCE and self.target != 'dx_depth1_unique':
                    loss = self.criterion(y_pred, item_target.unsqueeze(1).float().to(self.device))
                else:
                    loss = self.criterion(y_pred, item_target.float().to(self.device))
                avg_eval_loss += loss.item() / len(self.valid_dataloader)

                probs_eval = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_eval += list(probs_eval.flatten())
                truths_eval += list(item_target.detach().cpu().numpy().flatten())

            auroc_eval = roc_auc_score(truths_eval, preds_eval)
            auprc_eval = average_precision_score(truths_eval, preds_eval, average='micro')

        return avg_eval_loss, auroc_eval, auprc_eval

    def evaluation_both(self):
        self.model.eval()

        preds_eval = []
        truths_eval = []
        mimic_avg_eval_loss = 0.0

        with torch.no_grad():
            for iter, sample in enumerate(self.mimic_valid_dataloader):
                item_embed, item_target, seq_len = sample
                item_embed = item_embed.to(self.device)
                item_target = item_target.to(self.device)

                y_pred = self.model(item_embed, seq_len)

                if self.BCE and self.target != 'dx_depth1_unique':
                    loss = self.criterion(y_pred, item_target.unsqueeze(1).float().to(self.device))
                else:
                    loss = self.criterion(y_pred, item_target.float().to(self.device))
                mimic_avg_eval_loss += loss.item() / len(self.mimic_valid_dataloader)

                probs_eval = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_eval += list(probs_eval.flatten())
                truths_eval += list(item_target.detach().cpu().numpy().flatten())

            mimic_auroc_eval = roc_auc_score(truths_eval, preds_eval)
            mimic_auprc_eval = average_precision_score(truths_eval, preds_eval, average='micro')

            if not self.debug:
                wandb.log({'mimic_eval_loss': mimic_avg_eval_loss,
                           'mimic_eval_auroc': mimic_auroc_eval,
                           'mimic_eval_auprc': mimic_auprc_eval})

        preds_eval = []
        truths_eval = []
        eicu_avg_eval_loss = 0.0

        with torch.no_grad():
            for iter, sample in enumerate(self.eicu_valid_dataloader):
                item_embed, item_target, seq_len = sample
                item_embed = item_embed.to(self.device)
                item_target = item_target.to(self.device)

                y_pred = self.model(item_embed, seq_len)

                if self.BCE and self.target != 'dx_depth1_unique':
                    loss = self.criterion(y_pred, item_target.unsqueeze(1).float().to(self.device))
                else:
                    loss = self.criterion(y_pred, item_target.float().to(self.device))
                eicu_avg_eval_loss += loss.item() / len(self.eicu_valid_dataloader)

                probs_eval = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_eval += list(probs_eval.flatten())
                truths_eval += list(item_target.detach().cpu().numpy().flatten())

            eicu_auroc_eval = roc_auc_score(truths_eval, preds_eval)
            eicu_auprc_eval = average_precision_score(truths_eval, preds_eval, average='micro')

            if not self.debug:
                wandb.log({'eicu_eval_loss': eicu_avg_eval_loss,
                           'eicu_eval_auroc': eicu_auroc_eval,
                           'eicu_eval_auprc': eicu_auprc_eval})

        return mimic_avg_eval_loss, mimic_auroc_eval, mimic_auprc_eval, eicu_avg_eval_loss, eicu_auroc_eval, eicu_auprc_eval

    def test(self):
        ckpt = torch.load(self.best_eval_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        preds_test = []
        truths_test = []
        avg_test_loss = 0.

        with torch.no_grad():
            for iter, sample in enumerate(self.test_dataloader):
                item_id, item_target, seq_len = sample
                item_id = item_id.to(self.device)
                item_target = item_target.to(self.device)

                y_pred = self.model(item_id, seq_len)
                if self.BCE and self.target != 'dx_depth1_unique':
                    loss = self.criterion(y_pred, item_target.unsqueeze(1).float().to(self.device))
                else:
                    loss = self.criterion(y_pred, item_target.float().to(self.device))
                avg_test_loss += loss.item() / len(self.test_dataloader)

                probs_test = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_test += list(probs_test.flatten())
                truths_test += list(item_target.detach().cpu().numpy().flatten())

            auroc_test = roc_auc_score(truths_test, preds_test)
            auprc_test = average_precision_score(truths_test, preds_test, average='micro')

            if not self.debug:
                wandb.log({'test_loss': avg_test_loss,
                           'test_auroc': auroc_test,
                           'test_auprc': auprc_test})

                print('[Test]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_test_loss, auroc_test, auprc_test))

    def test_both(self):
        ckpt = torch.load(self.best_mimic_eval_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        preds_test = []
        truths_test = []
        avg_test_loss = 0.

        with torch.no_grad():
            for iter, sample in enumerate(self.mimic_test_dataloader):
                item_id, item_target, seq_len = sample
                item_id = item_id.to(self.device)
                item_target = item_target.to(self.device)

                y_pred = self.model(item_id, seq_len)
                if self.BCE and self.target != 'dx_depth1_unique':
                    loss = self.criterion(y_pred, item_target.unsqueeze(1).float().to(self.device))
                else:
                    loss = self.criterion(y_pred, item_target.float().to(self.device))
                avg_test_loss += loss.item() / len(self.mimic_test_dataloader)

                probs_test = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_test += list(probs_test.flatten())
                truths_test += list(item_target.detach().cpu().numpy().flatten())

            auroc_test = roc_auc_score(truths_test, preds_test)
            auprc_test = average_precision_score(truths_test, preds_test, average='micro')

            if not self.debug:
                wandb.log({'mimic_test_loss': avg_test_loss,
                           'mimic_test_auroc': auroc_test,
                           'mimic_test_auprc': auprc_test})

            print('[Test/mimic]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_test_loss, auroc_test,
                                                                                              auprc_test))

        ckpt = torch.load(self.best_eicu_eval_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        preds_test = []
        truths_test = []
        avg_test_loss = 0.

        with torch.no_grad():
            for iter, sample in enumerate(self.eicu_test_dataloader):
                item_id, item_target, seq_len = sample
                item_id = item_id.to(self.device)
                item_target = item_target.to(self.device)

                y_pred = self.model(item_id, seq_len)
                if self.BCE and self.target != 'dx_depth1_unique':
                    loss = self.criterion(y_pred, item_target.unsqueeze(1).float().to(self.device))
                else:
                    loss = self.criterion(y_pred, item_target.float().to(self.device))
                avg_test_loss += loss.item() / len(self.eicu_test_dataloader)

                probs_test = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_test += list(probs_test.flatten())
                truths_test += list(item_target.detach().cpu().numpy().flatten())

            auroc_test = roc_auc_score(truths_test, preds_test)
            auprc_test = average_precision_score(truths_test, preds_test, average='micro')

            if not self.debug:
                wandb.log({'eicu_test_loss': avg_test_loss,
                           'eicu_test_auroc': auroc_test,
                           'eicu_test_auprc': auprc_test})

            print('[Test/eicu]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_test_loss, auroc_test, auprc_test))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_induced', action='store_true')
    parser.add_argument('--source_file', choices=['mimic', 'eicu', 'both'], type=str)
    parser.add_argument('--few_shot', choices=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0], type=float)   # training_dataset_size ratio
    parser.add_argument('--target', choices=['readmission', 'mortality', 'los>3day', 'los>7day', 'dx_depth1_unique'], type=str)
    parser.add_argument('--item', choices=['lab', 'med', 'inf', 'all'], type=str)
    parser.add_argument('--max_length', type=str, default='150')
    parser.add_argument('--bert_model', choices=['bio_clinical_bert', 'bio_bert', 'pubmed_bert', 'blue_bert', 'bert', 'bert_mini', 'bert_small'], type=str)
    parser.add_argument('--bert_freeze', action='store_true')
    parser.add_argument('--path', type=str, default='/home/jylee/data/pretrained_ehr/output/KDD_output2/')
    parser.add_argument('--device_number', type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--concat', action='store_true')
    parser.add_argument('--cls_freeze', action='store_true')
    parser.add_argument('--only_BCE', action='store_true')
    parser.add_argument('--input_path', type=str, default='/home/jylee/data/pretrained_ehr/input_data/')
    parser.add_argument('--transformer', action='store_true')
    parser.add_argument('--transformer_layers', type=int, default=2)
    parser.add_argument('--transformer_attn_heads', type=int, default=8)
    parser.add_argument('--transformer_hidden_dim', type=int, default=256)
    parser.add_argument('--wandb_project_name', type=str)
    args = parser.parse_args()

    args.time_window = '12'
    args.rnn_model_type = 'gru'
    args.batch_size = 512
    args.rnn_bidirection = False
    args.n_epochs = 1000
    args.word_max_length = 15   # tokenized word max_length, used in padding
    # hyperparameter tuning
    args.dropout = 0.3
    args.embedding_dim = 128
    args.hidden_dim = 256
    args.lr = 1e-4

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_number)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if args.item == 'all':
    #     assert args.max_length == '300', 'when using all items, max length should be 300'

    mp.set_sharing_strategy('file_system')

    SEED = [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029]

    for seed in SEED:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

        args.seed = seed

        train_loader = get_dataloader(args=args, data_type='train')

        Trainer = DataSize_Trainer(args, train_loader, device)

        Trainer.train()


if __name__ == '__main__':
    main()