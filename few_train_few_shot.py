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

from models.rnn_bert_dict import dict_post_RNN
from models.rnn_models import RNNmodels
from utils.trainer_utils import *


def get_dataloader(args, data_type, data_name):
    dataset = FewTrainDataset(args, data_type, data_name)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    return dataloader


class FewTrainDataset(Dataset):
    def __init__(self, args, data_type, data_name):
        self.target = args.target
        item = args.item
        self.max_length = args.max_length
        time_window = args.time_window
        self.bert_induced = args.bert_induced
        few_shot = args.few_shot

        if few_shot == 1.0:
            path = os.path.join(args.input_path[:-1], item, '{}_{}_{}_{}_{}.pkl'.format(data_name, time_window, item, self.max_length, args.seed))
        else:
            path = os.path.join(args.input_path[:-1], item, '{}_{}_{}_{}_{}_{}.pkl'.format(data_name, time_window, item, self.max_length, args.seed, int(few_shot * 100)))

        data = pickle.load(open(path, 'rb'))

        if data_name == 'mimic':
            data = data.rename({'HADM_ID': 'ID'}, axis='columns')
        elif data_name == 'eicu':
            data = data.rename({'patientunitstayid': 'ID'}, axis='columns')

        self.item_name, self.item_target = self.preprocess(data, data_type, item, time_window, self.target)

        vocab_path = os.path.join(args.input_path + 'embed_vocab_file', item, 'both_{}_{}_{}_word2embed.pkl'.format(item, time_window, args.bert_model))
        self.id_dict = pickle.load(open(vocab_path, 'rb'))

    def __len__(self):
        return len(self.item_name)

    def __getitem__(self, item):
        single_target = self.item_target[item]
        if self.target == 'dx_depth1_unique':
            single_target = [int(j) for j in single_target]
            target_list = torch.Tensor(single_target).long()

            single_target = torch.zeros(18)
            single_target[target_list - 1] = 1  # shape of 18

        single_item_name = self.item_name[item]
        seq_len = torch.Tensor([len(single_item_name)])
        embedding = []

        def embed_dict(x):
            return self.id_dict[x]

        embedding = list(map(embed_dict, single_item_name))  # list with length seq_len
        embedding = torch.Tensor(embedding)

        padding = torch.zeros(int(self.max_length) - embedding.size(0))
        embedding = torch.cat((embedding, padding), dim=-1)

        return embedding, single_target, seq_len

    def preprocess(self, cohort, data_type, item, time_window, target):
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

        # drop with null item
        cohort = cohort[cohort.astype(str)[name_window] != '[]']

        # pad
        item_name = cohort[name_window].values.tolist()

        # target
        if target == 'dx_depth1_unique':
            item_target = cohort[target].values.tolist()

        else:
            item_target = torch.LongTensor(cohort[target].values.tolist())  # shape of (B)

        return item_name, item_target


class FewTrain():
    def __init__(self, args, device):
        self.source_train_dataloader = get_dataloader(args, data_type='train', data_name=args.source_file)
        self.source_eval_dataloader = get_dataloader(args, data_type='eval', data_name=args.source_file)
        self.source_test_dataloader = get_dataloader(args, data_type='test', data_name=args.source_file)

        self.test_train_dataloader = get_dataloader(args, data_type='train', data_name=args.test_file)
        self.test_eval_dataloader = get_dataloader(args, data_type='eval', data_name=args.test_file)
        self.test_test_dataloader = get_dataloader(args, data_type='test', data_name=args.test_file)

        self.device = device
        self.debug = args.debug
        self.BCE = args.only_BCE
        self.target = args.target
        lr = args.lr
        self.n_epochs = args.n_epochs
        self.source_file = args.source_file
        self.test_file = args.test_file

        if not self.debug:
            wandb.init(project=args.wandb_project_name, entity='pretrained_ehr', config=args, reinit=True)

        self.criterion = nn.BCEWithLogitsLoss()
        if args.target == 'dx_depth1_unique':
            output_size = 18
        else:
            output_size = 1

        file_target_name = args.target
        if file_target_name == 'los>3day':
            file_target_name = 'los_3days'
        elif file_target_name == 'los>7day':
            file_target_name = 'los_7days'

        if args.bert_induced:
            model_directory = 'cls_learnable'
            self.model = dict_post_RNN(args=args, output_size=output_size, device=self.device, target_file='both').to(self.device)
            print('cls_learnable RNN')
            source_filename = 'trained_clslearnable_from{}_dataportion{}_{}_best_auprc.pt'.format(args.source_file, args.few_shot, args.seed)
            test_filename = 'trained_clslearnable_to{}_dataportion{}_{}_best_auprc.pt'.format(args.test_file, args.few_shot, args.seed)
        else:
            model_directory = 'singleRNN'
            if args.item == 'lab':
                vocab_size = 14371 if args.concat else 448
            elif args.item == 'med':
                vocab_size = 4898 if args.concat else 2812
            elif args.item == 'inf':
                vocab_size = 979
            elif args.item == 'all':
                vocab_size = 15794 if args.concat else 3672

            self.model = RNNmodels(args, vocab_size, output_size, self.device).to(device)
            print('single rnn')
            source_filename = 'trained_single_rnn_from{}_dataportion{}_{}_best_auprc.pt'.format(args.source_file, args.few_shot, args.seed)
            test_filename = 'trained_single_rnn_to{}_dataportion{}_{}_best_auprc.pt'.format(args.test_file, args.few_shot, args.seed)

        self.source_path = os.path.join(args.path, args.item, model_directory, 'both', file_target_name, source_filename)
        self.test_path = os.path.join(args.path, args.item, model_directory, 'both', file_target_name, test_filename)
        print('source file path: {}'.format(self.source_path))
        print('test file path: {}'.format(self.test_path))


        self.source_optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.test_optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.source_early_stopping = EarlyStopping(patience=50, verbose=True)
        self.test_early_stopping = EarlyStopping(patience=50, verbose=True)

    def train(self):
        self.train_source()
        ckpt = torch.load(self.source_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        print('Model successfully loaded!')

        best_loss = float('inf')
        best_auroc = 0.0
        best_auprc = 0.0

        for n_epoch in range(self.n_epochs + 1):
            preds_train = []
            truths_train = []
            avg_train_loss = 0.

            for iter, sample in tqdm.tqdm(enumerate(self.test_train_dataloader)):
                self.model.train()
                self.test_optimizer.zero_grad(set_to_none=True)

                item_id, item_target, seq_len = sample
                item_id = item_id.to(self.device)
                item_target = item_target.to(self.device)

                y_pred = self.model(item_id, seq_len)
                if self.BCE and self.target != 'dx_depth1_unique':
                    loss = self.criterion(y_pred, item_target.unsqueeze(1).float().to(self.device))
                else:
                    loss = self.criterion(y_pred, item_target.float().to(self.device))

                loss.backward()
                self.test_optimizer.step()

                avg_train_loss += loss.item() / len(self.test_train_dataloader)

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
                                'optimizer_state_dict': self.test_optimizer.state_dict(),
                                'loss': best_loss,
                                'auroc': best_auroc,
                                'auprc': best_auprc,
                                'epochs': n_epoch}, self.test_path)
                print('Model parameter saved at epoch {}'.format(n_epoch))

            if not self.debug:
                wandb.log({'test_train_loss': avg_train_loss,
                           'test_train_auroc': auroc_train,
                           'test_train_auprc': auprc_train,
                           'test_eval_loss': avg_eval_loss,
                           'test_eval_auroc': auroc_eval,
                           'test_eval_auprc': auprc_eval})

            print('[Test/Train]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_train_loss, auroc_train, auprc_train))
            print('[Test/Valid]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_eval_loss, auroc_eval, auprc_eval))

            self.test_early_stopping(auprc_eval)
            if self.test_early_stopping.early_stop:
                print('Early stopping')
                break
        self.test()


    def train_source(self):
        best_loss = float('inf')
        best_auroc = 0.0
        best_auprc = 0.0

        for n_epoch in range(self.n_epochs + 1):
            preds_train = []
            truths_train = []
            avg_train_loss = 0.

            for iter, sample in tqdm.tqdm(enumerate(self.source_train_dataloader)):
                self.model.train()
                self.source_optimizer.zero_grad(set_to_none=True)

                item_id, item_target, seq_len = sample
                item_id = item_id.to(self.device)
                item_target = item_target.to(self.device)

                y_pred = self.model(item_id, seq_len)
                if self.BCE and self.target != 'dx_depth1_unique':
                    loss = self.criterion(y_pred, item_target.unsqueeze(1).float().to(self.device))
                else:
                    loss = self.criterion(y_pred, item_target.float().to(self.device))

                loss.backward()
                self.source_optimizer.step()

                avg_train_loss += loss.item() / len(self.source_train_dataloader)

                probs_train = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_train += list(probs_train.flatten())
                truths_train += list(item_target.detach().cpu().numpy().flatten())

            auroc_train = roc_auc_score(truths_train, preds_train)
            auprc_train = average_precision_score(truths_train, preds_train, average='micro')

            avg_eval_loss, auroc_eval, auprc_eval = self.evaluation_source()

            if best_auprc < auprc_eval:
                best_loss = avg_eval_loss
                best_auroc = auroc_eval
                best_auprc = auprc_eval
                if not self.debug:
                    torch.save({'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.source_optimizer.state_dict(),
                                'loss': best_loss,
                                'auroc': best_auroc,
                                'auprc': best_auprc,
                                'epochs': n_epoch}, self.source_path)
                print('Model parameter saved at epoch {}'.format(n_epoch))

            if not self.debug:
                wandb.log({'source_train_loss': avg_train_loss,
                           'source_train_auroc': auroc_train,
                           'source_train_auprc': auprc_train,
                           'source_eval_loss': avg_eval_loss,
                           'source_eval_auroc': auroc_eval,
                           'source_eval_auprc': auprc_eval})

            print('[Source/Train]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_train_loss, auroc_train, auprc_train))
            print('[Source/Valid]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_eval_loss, auroc_eval, auprc_eval))

            self.test_early_stopping(auprc_eval)
            if self.source_early_stopping.early_stop:
                print('===================Early stopping======================')
                print('===========Finished Learning Source File: {}==========='.format(self.source_file))
                break

    def evaluation_source(self):
        self.model.eval()

        preds_eval = []
        truths_eval = []
        avg_eval_loss = 0.

        with torch.no_grad():
            for iter, sample in enumerate(self.source_eval_dataloader):
                item_id, item_target, seq_len = sample
                item_id = item_id.to(self.device)
                item_target = item_target.to(self.device)

                y_pred = self.model(item_id, seq_len)
                if self.BCE and self.target != 'dx_depth1_unique':
                    loss = self.criterion(y_pred, item_target.unsqueeze(1).float().to(self.device))
                else:
                    loss = self.criterion(y_pred, item_target.float().to(self.device))
                avg_eval_loss += loss.item() / len(self.source_eval_dataloader)

                probs_eval = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_eval += list(probs_eval.flatten())
                truths_eval += list(item_target.detach().cpu().numpy().flatten())

            auroc_eval = roc_auc_score(truths_eval, preds_eval)
            auprc_eval = average_precision_score(truths_eval, preds_eval, average='micro')

        return avg_eval_loss, auroc_eval, auprc_eval

    def evaluation(self):
        self.model.eval()

        preds_eval = []
        truths_eval = []
        avg_eval_loss = 0.

        with torch.no_grad():
            for iter, sample in enumerate(self.test_eval_dataloader):
                item_id, item_target, seq_len = sample
                item_id = item_id.to(self.device)
                item_target = item_target.to(self.device)

                y_pred = self.model(item_id, seq_len)
                if self.BCE and self.target != 'dx_depth1_unique':
                    loss = self.criterion(y_pred, item_target.unsqueeze(1).float().to(self.device))
                else:
                    loss = self.criterion(y_pred, item_target.float().to(self.device))
                avg_eval_loss += loss.item() / len(self.test_eval_dataloader)

                probs_eval = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_eval += list(probs_eval.flatten())
                truths_eval += list(item_target.detach().cpu().numpy().flatten())

            auroc_eval = roc_auc_score(truths_eval, preds_eval)
            auprc_eval = average_precision_score(truths_eval, preds_eval, average='micro')

        return avg_eval_loss, auroc_eval, auprc_eval

    def test(self):
        ckpt = torch.load(self.test_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        preds_test = []
        truths_test = []
        avg_test_loss = 0.

        with torch.no_grad():
            for iter, sample in enumerate(self.test_test_dataloader):
                item_id, item_target, seq_len = sample
                item_id = item_id.to(self.device)
                item_target = item_target.to(self.device)

                y_pred = self.model(item_id, seq_len)
                if self.BCE and self.target != 'dx_depth1_unique':
                    loss = self.criterion(y_pred, item_target.unsqueeze(1).float().to(self.device))
                else:
                    loss = self.criterion(y_pred, item_target.float().to(self.device))
                avg_test_loss += loss.item() / len(self.test_test_dataloader)

                probs_test = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_test += list(probs_test.flatten())
                truths_test += list(item_target.detach().cpu().numpy().flatten())

            auroc_test = roc_auc_score(truths_test, preds_test)
            auprc_test = average_precision_score(truths_test, preds_test, average='micro')

            if not self.debug:
                wandb.log({'test_test_loss': avg_test_loss,
                           'test_test_auroc': auroc_test,
                           'test_test_auprc': auprc_test})

            print('[Test]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_test_loss, auroc_test, auprc_test))





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_induced', action='store_true')
    parser.add_argument('--source_file', choices=['mimic', 'eicu'])
    parser.add_argument('--test_file', choices=['mimic', 'eicu'])
    parser.add_argument('--few_shot', type=float, choices=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], default=0.0)
    parser.add_argument('--target', choices=['readmission', 'mortality', 'los>3day', 'los>7day', 'dx_depth1_unique'], type=str, default='readmission')
    parser.add_argument('--item', choices=['lab', 'med', 'inf', 'all'], type=str, default='med')
    parser.add_argument('--time_window', choices=['12', '24', '36', '48', 'Total'], type=str, default='12')
    parser.add_argument('--rnn_model_type', choices=['gru', 'lstm'], type=str, default='gru')
    parser.add_argument('--batch_size', type=int, default=512)
    # parser.add_argument('--dropout', type=float, default=0.1)
    # parser.add_argument('--embedding_dim', type=int, default=768)
    # parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--rnn_bidirection', action='store_true')
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_length', type=str, default='150')
    parser.add_argument('--bert_model',choices=['bio_clinical_bert', 'bio_bert', 'pubmed_bert', 'blue_bert', 'bert_mini', 'bert_tiny'], type=str, default='bio_bert')
    parser.add_argument('--bert_freeze', action='store_true')
    parser.add_argument('--input_path', type=str, default='/home/jylee/data/pretrained_ehr/input_data/')
    parser.add_argument('--path', type=str, default='/home/jylee/data/pretrained_ehr/output/KDD_output/')
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

    assert args.source_file != args.test_file, 'source file and test file should be different'

    args.rnn_bidirection = False

    args.dropout = 0.3
    args.embedding_dim = 128
    args.hidden_dim = 256

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
        print(seed)

        trainer = FewTrain(args, device)
        trainer.train()


if __name__ == '__main__':
    main()