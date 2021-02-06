import torch
import torch.nn as nn
import torch.multiprocessing as mp

from sklearn.metrics import roc_auc_score, average_precision_score

import random
import numpy as np
import wandb
import os
import argparse
import tqdm

from dataset.prebert_dict_dataloader import bertinduced_dict_get_dataloader
from models.transformer import Transformer
from models.rnn_models import RNNmodels
from models.rnn_bert_dict import dict_post_RNN
from utils.trainer_utils import EarlyStopping


class FinetuningBoth():
    def __init__(self, args, device):
        target_file = args.target_file
        assert args.source_file == 'both', 'finetuning should be conducted at both'
        assert target_file == 'mimic' or target_file=='eicu', 'target file should be either mimic or eicu'

        self.both_train_dataloader = bertinduced_dict_get_dataloader(args, data_type='train', data_name='both')

        self.dataloader = bertinduced_dict_get_dataloader(args, data_type='train', data_name=target_file)
        self.valid_dataloader = bertinduced_dict_get_dataloader(args, data_type='eval', data_name=target_file)
        self.test_dataloader = bertinduced_dict_get_dataloader(args, data_type='test', data_name=target_file)

        self.device = device
        self.debug = args.debug
        self.BCE = args.only_BCE
        self.target = args.target
        self.source_file = args.source_file    # both
        self.lr_scheduler = args.lr_scheduler

        if not self.debug:
            wandb.init(project=args.wandb_project_name, entity='pretrained_ehr', config=args, reinit=True)

        lr = args.lr
        self.lr = lr
        self.n_epochs = args.n_epochs
        self.pretrain_epochs = args.pretrain_epochs

        file_target_name = args.target
        if file_target_name == 'los>3day':
            file_target_name = 'los_3days'
        elif file_target_name == 'los>7day':
            file_target_name = 'los_7days'

        if not args.transformer:
            assert not args.cls_freeze and not args.concat and args.only_BCE, 'no cls_freeze, no concat, only BCE'
            if args.bert_induced:
                filename = 'cls_learnable_finetune_both_rnn_{}_{}_{}_{}_onlyBCE'.format(target_file, args.bert_model, args.seed, args.pretrain_epochs)  # adjust this
                pretrained_filename = 'cls_learnable_pretrained_both_rnn_{}_{}_{}'.format(args.bert_model, args.seed, args.pretrain_epochs)
            else:
                filename = 'single_finetune_both_rnn_{}_{}_{}_{}_onlyBCE'.format(target_file, args.bert_model, args.seed, args.pretrain_epochs)
                pretrained_filename = 'single_pretrained_both_rnn_{}_{}_{}'.format(args.bert_model, args.seed, args.pretrain_epochs)
        elif args.transformer:
            raise NotImplementedError

        if args.bert_induced:
            finetune_path = os.path.join(args.path, args.item, 'cls_learnable', args.source_file, file_target_name, filename)
            pretrain_path = os.path.join(args.path, args.item, 'cls_learnable', args.source_file, file_target_name, pretrained_filename)
        elif not args.bert_induced:
            finetune_path = os.path.join(args.path, args.item, 'singleRNN', args.source_file, file_target_name, filename)
            pretrain_path = os.path.join(args.path, args.item, 'singleRNN', args.source_file, file_target_name, pretrained_filename)
        print('Pretrained model will be saved in {}'.format(pretrain_path))
        print('')
        print('Finetuned model will be saved in {}'.format(finetune_path))

        self.best_pretrain_path = pretrain_path + '_best_auprc.pt'
        self.best_eval_path = finetune_path + '_best_auprc.pt'
        self.final_path = finetune_path + '_final.pt'

        self.criterion = nn.BCEWithLogitsLoss()
        if args.target == 'dx_depth1_unique':
            output_size = 18
        else:
            output_size = 1

        if args.bert_induced:
            print('bert-induced RNN')
            self.model = dict_post_RNN(args=args, output_size=output_size, device=self.device, target_file='both').to(self.device)
        elif not args.bert_induced:
            print('singleRNN')
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
            self.model = RNNmodels(args=args, vocab_size=vocab_size, output_size=output_size, device=device).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        if args.lr_scheduler is not None:
            if args.lr_scheduler == 'labmda30':
                lambda1 = lambda epoch: 0.95 ** (epoch/30)
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=[lambda1])

            elif args.lr_scheduler == 'lambda20':
                lambda1 = lambda epoch: 0.90 ** (epoch/20)
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=[lambda1])

            elif args.lr_scheduler == 'plateau':
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', min_lr=1e-6)

        self.pretrain_early_stopping = EarlyStopping(patience=50, verbose=True)
        self.early_stopping = EarlyStopping(patience=50, verbose=True)

    def train(self):
        ckpt = torch.load(self.best_pretrain_path)
        self.model.load_state_dict(ckpt['model_state_dict'])

        best_loss = float('inf')
        best_auroc = 0.0
        best_auprc = 0.0

        for n_epoch in range(self.n_epochs + 1):
            preds_train = []
            truths_train = []
            avg_train_loss = 0.0

            for iter, sample in tqdm.tqdm(enumerate(self.dataloader)):
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                item_embed, item_target, seq_len = sample
                item_embed = item_embed.to(self.device)
                item_target = item_target.to(self.device)
                y_pred = self.model(item_embed, seq_len)

                if self.BCE and self.target != 'dx_depth1_unique':
                    loss = self.criterion(y_pred, item_target.unsqueeze(1).float().to(self.device))
                else:
                    loss = self.criterion(y_pred, item_target.float().to(self.device))
                loss.backward()
                self.optimizer.step()

                avg_train_loss += loss.item() / len(self.dataloader)

                probs_train = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_train += list(probs_train.flatten())
                truths_train += list(item_target.detach().cpu().numpy().flatten())

            auroc_train = roc_auc_score(truths_train, preds_train)
            auprc_train = average_precision_score(truths_train, preds_train, average='micro')

            avg_eval_loss, auroc_eval, auprc_eval = self.evaluation()

            if best_auprc < auprc_eval:
                best_loss = avg_eval_loss
                best_auroc = auroc_eval
                best_auprc = auprc_eval
                if not self.debug:
                    torch.save({'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': best_loss,
                                'auorc': best_auroc,
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


            print('[Train]  loss: {:.3f},  auroc: {:.3f},   auprc: {:.3f}'.format(avg_train_loss, auroc_train,
                                                                                  auprc_train))
            print('[Valid]  loss: {:.3f},  auroc: {:.3f},   auprc: {:.3f}'.format(avg_eval_loss, auroc_eval,
                                                                                  auprc_eval))

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
                break

        self.test()

    def evaluation(self):
        self.model.eval()

        preds_eval = []
        truths_eval = []
        avg_eval_loss = 0.0

        with torch.no_grad():
            for iter, sample in enumerate(self.valid_dataloader):
                item_embed, item_target, seq_len = sample
                item_embed = item_embed.to(self.device)
                item_target = item_target.to(self.device)
                y_pred = self.model(item_embed, seq_len)

                if self.BCE and self.target != 'dx_depth1_unique':
                    loss = self.criterion(y_pred, item_target.unsqueeze(1).float().to(self.deivce))
                else:
                    loss = self.criterion(y_pred, item_target.float().to(self.device))
                avg_eval_loss += loss.item() / len(self.valid_dataloader)

                probs_eval = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_eval += list(probs_eval.flatten())
                truths_eval += list(item_target.detach().cpu().numpy().flatten())

            auroc_eval = roc_auc_score(truths_eval, preds_eval)
            auprc_eval = average_precision_score(truths_eval, preds_eval, average='micro')

        return avg_eval_loss, auroc_eval, auprc_eval

    def test(self):
        ckpt = torch.load(self.best_eval_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        preds_test = []
        truths_test = []
        avg_test_loss = 0.

        with torch.no_grad():
            for iter, sample in enumerate(self.test_dataloader):
                item_embed, item_target, seq_len = sample
                item_embed = item_embed.to(self.device)
                item_target = item_target.to(self.device)
                y_pred = self.model(item_embed, seq_len)
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



    def pretrain(self):
        best_auprc = 0.0
        for n_epoch in range(self.pretrain_epochs + 1):
            preds_train = []
            truths_train = []
            avg_train_loss = 0.

            for iter, sample in tqdm.tqdm(enumerate(self.both_train_dataloader)):
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

                avg_train_loss += loss.item() / len(self.both_train_dataloader)

                probs_train = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_train += list(probs_train.flatten())
                truths_train += list(item_target.detach().cpu().numpy().flatten())

            auroc_train = roc_auc_score(truths_train, preds_train)
            auprc_train = average_precision_score(truths_train, preds_train, average='micro')

            if best_auprc < auprc_train:
                best_auprc = auprc_train
                if not self.debug:
                    torch.save({'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'auprc': best_auprc}, self.best_pretrain_path)

                    wandb.log({'pretrain_train_loss': avg_train_loss,
                               'pretrain_train_auroc': auroc_train,
                               'pretrain_train_auprc': auprc_train})

                print('Model parameter saved at epoch {}'.format(n_epoch))

            print('[Train]  loss: {:.3f},  auroc: {:.3f},   auprc: {:.3f}'.format(avg_train_loss, auroc_train, auprc_train))

            self.pretrain_early_stopping(auprc_train)
            if self.pretrain_early_stopping.early_stop:
                print('Early Stopping')
                break
        print('==========================================')
        print('Finished pretraining!')
        print('==========================================')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_induced', action='store_true')
    parser.add_argument('--source_file', choices=['both'], type=str, default='both')
    parser.add_argument('--target_file', choices=['mimic', 'eicu'], type=str)
    parser.add_argument('--target', choices=['readmission', 'mortality', 'los>3day', 'los>7day', 'dx_depth1_unique'])
    parser.add_argument('--item', choices=['lab', 'med', 'inf', 'all'], type=str, default='all')
    parser.add_argument('--task', choices=['pretrain', 'finetune'], type=str)
    parser.add_argument('--time_window', choices=['12', '24', '36', '48', 'Total'], type=str, default='12')
    parser.add_argument('--rnn_model_type', choices=['gru', 'lstm'], type=str, default='gru')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--rnn_bidirection', action='store_true')
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--pretrain_epochs', type=int)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_length', type=str, default='150')
    parser.add_argument('--bert_model', choices=['bert', 'bio_clinical_bert', 'bio_bert', 'pubmed_bert', 'blue_bert', 'bert_mini', 'bert_tiny'], type=str, default='bio_bert')
    parser.add_argument('--bert_freeze', action='store_true')
    parser.add_argument('--input_path', type=str, default='/home/jylee/data/pretrained_ehr/input_data/')
    parser.add_argument('--path', type=str, default='/home/jylee/data/pretrained_ehr/output/KDD_output/')
    parser.add_argument('--word_max_length', type=int, default=15)  # tokenized word max_length, used in padding
    parser.add_argument('--device_number', type=str, default='7')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--concat', action='store_true')
    parser.add_argument('--cls_freeze', action='store_true')
    parser.add_argument('--only_BCE', action='store_true')
    parser.add_argument('--transformer', action='store_true')
    parser.add_argument('--wandb_project_name', type=str)
    parser.add_argument('--lr_scheduler', choices=['lambda30', 'lambda20', 'plateau'])
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_number)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.rnn_bidirection = False
    args.dropout = 0.3
    args.embedding_dim = 128
    args.hidden_dim = 256

    mp.set_sharing_strategy('file_system')

    SEED = [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029]

    for seed in SEED:
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

        args.seed = seed

        if args.task == 'finetune':
            print('Finetuning on {}'.format(args.target_file))
            trainer = FinetuningBoth(args, device)
            trainer.train()

        elif args.task == 'pretrain':
            print('Pretrain')
            trainer = FinetuningBoth(args, device)
            trainer.pretrain()

        print('Finished training seed: {}'.format(seed))


if __name__ == '__main__':
    main()
