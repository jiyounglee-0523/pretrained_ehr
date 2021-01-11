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
import re

from models.rnn_bert_dict import *
from models.rnn_models import *
from models.prebert import *
from utils.loss import *
from utils.trainer_utils import *

"""
few-shot: 0.0(zero-shot), 0.1, 0.3, 0.5, 0.7, 0.9, 1.0(full-shot = transfer learning)
"""


def get_test_dataloader(args, data_type='train'):       # validation? test?
    if data_type == 'train':
        train_data = Few_Shot_Dataset(args, data_type=data_type)
        dataloader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=32)

    elif data_type == 'eval':
        eval_data = Few_Shot_Dataset(args, data_type=data_type)
        dataloader = DataLoader(dataset=eval_data, batch_size=args.batch_size, shuffle=True, num_workers=32)

    elif data_type == 'test':
        test_data = Few_Shot_Dataset(args, data_type=data_type)
        dataloader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, num_workers=32)

    return dataloader


class Few_Shot_Dataset(Dataset):
    def __init__(self, args, data_type):
        test_file = args.test_file
        self.target = args.target
        item = args.item
        self.max_length = args.max_length
        time_window = args.time_window
        self.word_max_length = args.word_max_length
        self.bert_induced = args.bert_induced

        if test_file == 'both':
            raise NotImplementedError

        else:
            few_shot = args.few_shot
            if few_shot == 0.0 or few_shot == 1.0:
                path = '/home/jylee/data/pretrained_ehr/input_data/{}_{}_{}_{}_{}.pkl'.format(test_file, time_window,
                                                                                                 item, self.max_length,
                                                                                                 args.seed)
            else:
                path = '/home/jylee/data/pretrained_ehr/input_data/{}_{}_{}_{}_{}_{}.pkl'.format(test_file, time_window, item, self.max_length, args.seed, int(few_shot * 100))
            data = pickle.load(open(path, 'rb'))

            # change column name
            if test_file == 'mimic':
                data = data.rename({'HADM_ID': 'ID'}, axis='columns')

            elif test_file == 'eicu':
                data = data.rename({'patientunitstayid': 'ID'}, axis='columns')

            self.item_name, self.item_id, self.item_offset, self.item_offset_order, self.item_target = self.preprocess(data, data_type, item, time_window, self.target)

            vocab_path = os.path.join('/home/jylee/data/pretrained_ehr', '{}_{}_id_dict.pkl'.format(test_file, item))
            self.word2embed = pickle.load(open(vocab_path, 'rb'))

    def __len__(self):
        return self.item_id.size(0)

    def __getitem__(self, item):
        # RNN
        single_item_id = self.item_id[item]
        single_item_offset = self.item_offset[item]
        single_item_offset_order = self.item_offset_order[item]
        single_target = self.item_target[item]
        single_length = torch.LongTensor([torch.max(torch.nonzero(single_item_id.data)) + 1])

        if self.target == 'dx_depth1_unique':
            single_target = [int(j) for j in single_target]
            target_list = torch.Tensor(single_target).long()

            single_target = torch.zeros(18)
            single_target[target_list - 1] = 1   # shape of 18

        # bert_induced
        single_item_name = self.item_name[item]
        seq_len = torch.Tensor([len(single_item_name)])

        single_item_name = [re.sub(r'[.,/|!-?"\':;~()\[\]]', '', i) for i in single_item_name]

        def embed_dict(x):
            return self.word2embed[x]
        embedding = list(map(embed_dict, single_item_name))  # list with length seq_len
        embedding = torch.Tensor(embedding)

        padding = torch.zeros(int(self.max_length) - embedding.size(0))
        embedding = torch.cat((embedding, padding), dim=-1)

        if not self.bert_induced:
            return single_item_id, single_target, single_length
        elif self.bert_induced:
            return embedding, single_target, seq_len


    def preprocess(self, cohort, data_type, item, time_window, target):
        if time_window == 'Total':
            name_window = '{}_name'.format(item)
            offset_window = 'order_offset'
            offset_order_window = '{}_offset_order'.format(item)
            id_window = '{}_id_{}hr'.format(item, time_window)
            target_fold = '{}_fold'.format(target)

        else:
            name_window = '{}_name_{}hr'.format(item, time_window)
            offset_window = 'order_offset_{}hr'.format(time_window)
            offset_order_window = '{}_offset_order_{}hr'.format(item, time_window)
            id_window = '{}_id_{}hr'.format(item, time_window)
            target_fold = '{}_fold'.format(target)
            if target == 'dx_depth1_unique':
                target_fold = 'dx_fold'

        # extract cohort
        cohort = cohort[['ID', name_window, offset_window, offset_order_window, id_window, target, target_fold]]
        cohort = cohort[cohort[target_fold] != -1]   # -1 is for unsampled

        if data_type == 'train':
            cohort = cohort[cohort[target_fold] == 1]

        elif data_type == 'eval':
            cohort = cohort[cohort[target_fold] == 2]

        elif data_type == 'test':
            cohort = cohort[cohort[target_fold] == 0]

        # pad
        item_name = cohort[name_window].values.tolist()

        item_id = cohort[id_window].apply(lambda x: torch.Tensor(x)).values.tolist()
        item_id = pad_sequence(item_id, batch_first=True)

        item_offset_order = cohort[offset_order_window].apply(lambda x: torch.Tensor(x)).values.tolist()
        item_offset_order = pad_sequence(item_offset_order, batch_first=True)

        item_offset = cohort[offset_window].apply(lambda x: torch.Tensor(x)).values.tolist()
        item_offset = pad_sequence(item_offset, batch_first=True)

        # target
        if target == 'dx_depth1_unique':
            item_target = cohort[target].values.tolist()

        else:
            item_target = torch.LongTensor(cohort[target].values.tolist()) # shape of (B)

        return item_name, item_id, item_offset, item_offset_order, item_target


################################################################################


class Tester(nn.Module):
    def __init__(self, args, train_dataloader, valid_dataloader, test_dataloader, device, seed):
        super().__init__()
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.device = device

        wandb.init(project='test_learnable_cls_output', entity='pretrained_ehr', config=args, reinit=True)

        lr = args.lr
        self.n_epochs = args.n_epochs

        if args.target == 'dx_depth1_unique':
            output_size = 18
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            output_size = 1
            self.criterion = FocalLoss()

        file_target_name = args.target
        if file_target_name == 'los>3day':
            file_target_name = 'los_3days'
        elif file_target_name == 'los>7day':
            file_target_name = 'los_7days'


        if args.bert_induced and args.bert_freeze:
            model_directory = 'cls_learnable'
            self.model = dict_post_RNN(args=args, output_size=output_size, device=self.device).to(device)
            print('bert freeze')
            filename = 'cls_learnable_{}'.format(args.seed)
        elif args.bert_induced and not args.bert_freeze:
            model_directory = 'bert_finetune'
            self.model = post_RNN(args=args, output_size=output_size).to(device)
            print('bert finetuning')
        elif not args.bert_induced:
            model_directory = 'singleRNN'
            if args.source_file == 'mimic':
                vocab_size = 545
            elif args.source_file == 'eicu':
                vocab_size = 157
            else:
                raise NotImplementedError

            self.model = RNNmodels(args, vocab_size, output_size, self.device).to(device)
            print('single rnn')
            filename = 'trained_single_rnn_{}'.format(args.seed)

        self.source_path = os.path.join(args.path, model_directory, args.source_file, file_target_name, filename)

        target_filename = 'few_shot{}_from{}_to{}_seed{}'.format(args.few_shot, args.source_file, args.test_file,seed)
        target_path = os.path.join(args.path, model_directory, args.test_file, file_target_name, target_filename)


        self.best_target_path = target_path + '_best_auprc.pt'
        self.final_path = target_path + '_final.pt'


        # load parameters
        best_eval_path = self.source_path + '_best_auprc.pt'
        print('Load Model from {}'.format(best_eval_path))
        ckpt = torch.load(best_eval_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        print("Model successfully loaded!")
        print('Model will be saved in {}'.format(self.best_target_path))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.early_stopping = EarlyStopping(patience=30, verbose=True)
        num_params = count_parameters(self.model)
        print('Number of parameters: {}'.format(num_params))

    def train(self):
        best_loss = float('inf')
        best_auroc = 0.0
        best_auprc = 0.0

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
                loss = self.criterion(y_pred, item_target.float().to(self.device))

                loss.backward()
                self.optimizer.step()

                avg_train_loss += loss.item() / len(self.train_dataloader)

                probs_train = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_train += list(probs_train.flatten())
                truths_train += list(item_target.detach().cpu().numpy().flatten())

            auroc_train = roc_auc_score(truths_train, preds_train)
            auprc_train = average_precision_score(truths_train, preds_train)

            avg_eval_loss, auroc_eval, auprc_eval = self.evaluation()

            if best_auprc < auprc_eval:
                best_loss = avg_eval_loss
                best_auroc = auroc_eval
                best_auprc = auprc_eval

                torch.save({'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': best_loss,
                            'auroc': best_auroc,
                            'auprc': best_auprc,
                            'epochs': n_epoch}, self.best_target_path)
                print('Model parameter saved at epoch {}'.format(n_epoch))

            wandb.log({'train_loss': avg_train_loss,
                       'train_auroc': auroc_train,
                       'train_auprc': auprc_train,
                       'eval_loss': avg_eval_loss,
                       'eval_auroc': auroc_eval,
                       'eval_auprc': auprc_eval})

            print('[Train]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_train_loss, auroc_train,
                                                                                         auprc_train))
            print('[Valid]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_eval_loss, auroc_eval,
                                                                                         auprc_eval))

            self.early_stopping(auprc_eval)
            if self.early_stopping.early_stop:
                print('Early stopping')
                torch.save({'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': avg_eval_loss,
                            'auroc': best_auroc,
                            'auprc': best_auprc,
                            'epochs': n_epoch}, self.final_path)

                break

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
                loss = self.criterion(y_pred, item_target.float().to(self.device))
                avg_eval_loss += loss.item() / len(self.valid_dataloader)

                probs_eval = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_eval += list(probs_eval.flatten())
                truths_eval += list(item_target.detach().cpu().numpy().flatten())

            auroc_eval = roc_auc_score(truths_eval, preds_eval)
            auprc_eval = average_precision_score(truths_eval, preds_eval)

        return avg_eval_loss, auroc_eval, auprc_eval



    def test(self):
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
                loss = self.criterion(y_pred, item_target.float().to(self.device))
                avg_test_loss += loss.item() / len(self.test_dataloader)

                probs_test = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_test += list(probs_test.flatten())
                truths_test += list(item_target.detach().cpu().numpy().flatten())

            auroc_test = roc_auc_score(truths_test, preds_test)
            auprc_test = average_precision_score(truths_test, preds_test)

            wandb.log({'test_loss': avg_test_loss,
                       'test_auroc': auroc_test,
                       'test_auprc': auprc_test})

            print('[Test]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_test_loss, auroc_test,
                                                                                     auprc_test))

        return avg_test_loss, auroc_test, auprc_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_induced', action='store_true')
    parser.add_argument('--source_file', choices=['mimic', 'eicu', 'both'], type=str, default='both')
    parser.add_argument('--test_file', choices=['mimic', 'eicu', 'both'], type=str, default='eicu')
    parser.add_argument('--few_shot', type=float, choices=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], default=0.0)
    parser.add_argument('--target', choices=['readmission', 'mortality', 'los>3day', 'los>7day', 'dx_depth1_unique'], type=str, default='dx_depth1_unique')
    parser.add_argument('--item', choices=['lab', 'diagnosis', 'chartevent', 'medication', 'infusion'], type=str, default='lab')
    parser.add_argument('--time_window', choices=['12', '24', '36', '48', 'Total'], type=str, default='12')
    parser.add_argument('--rnn_model_type', choices=['gru', 'lstm'], type=str, default='gru')
    parser.add_argument('--batch_size', type=int, default=256)
    # parser.add_argument('--dropout', type=float, default=0.1)
    # parser.add_argument('--embedding_dim', type=int, default=768)
    # parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--rnn_bidirection', action='store_true')
    parser.add_argument('--n_epochs', type=int, default=500)
    # parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max_length', type=str, default='150')
    parser.add_argument('--bert_model', type=str, default='clinical_bert')
    parser.add_argument('--bert_freeze', action='store_true')
    parser.add_argument('--path', type=str, default='/home/jylee/data/pretrained_ehr/output/KDD_output/')
    parser.add_argument('--word_max_length', type=int, default=15)  # tokenized word max_length, used in padding
    parser.add_argument('--device_number', type=int, default=0)
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_number)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.rnn_bidirection = False
    args.bert_freeze = True

    if args.source_file == args.test_file:
        assert args.few_shot == 0.0, "there is no few_shot if source and test file are the same"

    # hyperparameter tuning
    args.dropout = 0.3
    args.embedding_dim = 128
    args.hidden_dim = 256
    args.lr = 1e-4

    mp.set_sharing_strategy('file_system')

    SEED = [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029]

    for seed in SEED:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        args.seed = seed

        train_loader = get_test_dataloader(args=args, data_type='train')
        valid_loader = get_test_dataloader(args=args, data_type='eval')
        test_loader = get_test_dataloader(args=args, data_type='test')

        tester = Tester(args, train_loader, valid_loader, test_loader, device, seed)

        if args.few_shot == 0.0:
            print('Only test')
            tester.test()
        else:
            print('Train then test')
            tester.train()
            tester.test()

if __name__ == '__main__':
    main()

