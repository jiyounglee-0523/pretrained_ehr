import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score

import wandb
import random
import numpy as np
import argparse
import os
import pickle
import re
import tqdm

from utils.trainer_utils import *

def get_dataloader(args, data_type):
    dataset = RNNbottomDataset(args, data_type)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    return dataloader

class RNNbottomDataset(Dataset):
    def __init__(self, args, data_type):
        self.target = args.target
        item = args.item
        self.max_length = args.max_length
        time_window = args.time_window
        self.word_max_length = args.word_max_length

        if args.few_shot == 0.0 or args.few_shot == 1.0:
            path = os.path.join(args.input_path[:-1], item, '{}_{}_{}_{}_{}.pkl'.format(args.test_file, time_window, item, self.max_length, args.seed))
        else:
            path = os.path.join(args.input_path[:-1], item, '{}_{}_{}_{}_{}_{}.pkl'.format(args.test_file, time_window, item, self.max_length, args.seed, int(args.few_shot * 100)))

        data = pickle.load(open(path, 'rb'))

        if args.test_file == 'mimic':
            data = data.rename({'HADM_ID': 'ID'}, axis='columns')

        elif args.test_file == 'eicu':
            data = data.rename({'patientunitstayid': 'ID'}, axis='columns')

        self.item_name, self.item_target = self.preprocess(data, data_type, item, time_window, self.target)

        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        id_dict_path = os.path.join(args.input_path[:-1], 'subword2index_{}_all.pkl'.format(args.test_file))
        self.id_dict = pickle.load(open(id_dict_path, 'rb'))

    def __len__(self):
        return len(self.item_name)

    def __getitem__(self, item):
        single_item_name = self.item_name[item]
        seq_len = torch.Tensor([len(single_item_name)])
        # pad
        pad_length = int(self.max_length) - len(single_item_name)
        padding = torch.zeros([pad_length, self.word_max_length])
        padding_seq = torch.ones([pad_length])

        def organize(x):
            return re.sub(r'[,|!?"\':;~()\[\]]', '', x)
        single_item_name = list(map(organize, single_item_name))
        single_item_name = self.tokenizer(single_item_name, padding='max_length', return_tensors='pt', max_length=self.word_max_length+2)
        single_item_name = single_item_name['input_ids']
        single_item_name = single_item_name.reshape(-1).tolist()
        single_item_name = list(filter(lambda x: x!=101 and x!=102, single_item_name))

        def embed_dict(x):
            return self.id_dict[x]
        embedding = list(map(embed_dict, single_item_name))
        embedding = torch.Tensor(embedding).reshape(-1, self.word_max_length)  # shape of (seq_len, word_max_length)
        word_seq_len = torch.argmin(embedding, dim=1)     # shape of (seq_len)

        # pad
        embedding = torch.cat((embedding, padding), dim=0)
        word_seq_len = torch.cat((word_seq_len, padding_seq), dim=0)

        # target
        single_target = self.item_target[item]
        if self.target == 'dx_depth1_unique':
            single_target = [int(j) for j in single_target]
            target_list = torch.Tensor(single_target).long()

            single_target = torch.zeros(18)
            single_target[target_list - 1] = 1  # shape of 18
        return embedding, single_target, seq_len, word_seq_len


    def preprocess(self, cohort, data_type, item, time_window, target):
        name_window = '{}_name_{}hr'.format(item, time_window)
        offset_window = 'order_offset_{}hr'.format(time_window)
        offset_order_window = '{}_offset_order_{}hr'.format(item, time_window)
        target_fold = '{}_fold'.format(target)
        if target == 'dx_depth1_unique':
            target_fold = 'dx_fold'

        # extract cohort
        cohort = cohort[['ID', name_window, offset_window, offset_order_window, target, target_fold]]

        if data_type == 'train':
            cohort = cohort[cohort[target_fold] == 1]
        elif data_type == 'eval':
            cohort = cohort[cohort[target_fold] == 2]
        elif data_type == 'test':
            cohort = cohort[cohort[target_fold] == 0]

        # drop with null item
        cohort = cohort[cohort.astype(str)[name_window] != '[]']

        # pad
        item_name = cohort[name_window].values.tolist()  # list of item_names

        if target == 'dx_depth1_unique':
            item_target = cohort[target].values.tolist()
        else:
            item_target = torch.LongTensor(cohort[target].values.tolist())  # shape of (B)

        return item_name, item_target


class RNNBottom(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.bottom_hidden_size = args.bottom_hidden_size
        self.word_max_length = args.word_max_length
        vocab_size = 1176 if args.test_file == 'mimic' else 813
        self.embedding = nn.Embedding(vocab_size, args.bottom_embedding_size)
        self.model = nn.GRU(args.bottom_embedding_size, args.bottom_hidden_size, num_layers=1, dropout=args.dropout, batch_first=True, bidirectional=True)

        self.linear_1 = nn.Linear(2 * args.bottom_hidden_size, args.embedding_dim)

    def forward(self, x, word_seq_len):
        x = x.reshape(-1, self.word_max_length)
        x = self.embedding(x.long().to(self.device))
        lengths = word_seq_len.reshape(-1).long()  # check!
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.model(packed)
        output_seq, output_len = pad_packed_sequence(output, batch_first=True)

        i = range(x.size(0))

        forward_output = output_seq[i, lengths-1, :self.bottom_hidden_size]
        backward_output = output_seq[:, 0, self.bottom_hidden_size:]
        output = torch.cat((forward_output, backward_output), dim=-1)
        output = self.linear_1(output)
        return output

class RNNtop(nn.Module):
    def __init__(self, args, output_size, device):
        super().__init__()
        self.device = device
        self.hidden_dim = args.hidden_dim
        self.embedding_dim = args.embedding_dim

        self.rnn_bottom = RNNBottom(args, device).to(device)
        self.model = nn.GRU(args.embedding_dim, args.hidden_dim, num_layers=1, dropout=args.dropout, batch_first=True, bidirectional=False)
        self.output_fc = nn.Linear(self.hidden_dim, output_size)

    def forward(self, x, lengths, word_seq_len):
        B = x.size(0)
        rnn_output = self.rnn_bottom(x, word_seq_len)  # shape of (batch_size X seq_len, embedding_dim)
        rnn_output = rnn_output.reshape(B, -1, self.embedding_dim)

        lengths = lengths.squeeze(-1).long()
        packed = pack_padded_sequence(rnn_output, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.model(packed)
        output_seq, output_len = pad_packed_sequence(output, batch_first=True)

        i = range(B)
        output = output_seq[i, lengths - 1, :]

        output = self.output_fc(output)
        return output


class RNNBottomTrainer():
    def __init__(self, args, device):
        self.train_dataloader = get_dataloader(args, data_type='train')
        self.eval_dataloader = get_dataloader(args, data_type='eval')
        self.test_dataloader = get_dataloader(args, data_type='test')

        self.device = device
        self.BCE = args.only_BCE
        self.target = args.target
        lr = args.lr
        self.n_epochs = args.n_epochs

        wandb.init(project= args.wandb_project_name, entity='pretrained_ehr', config=args, reinit=True)

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

        self.model = RNNtop(args, output_size, device).to(device)
        model_directory = 'bert_finetune'
        source_filename = 'finetuning_bottomrnn_{}_best_auprc.pt'.format(args.seed)
        source_best_path = os.path.join(args.path, args.item, model_directory, args.source_file, file_target_name, source_filename)
        print('Load path: {}'.format(source_best_path))

        target_filename = 'finetuning_fewshot{}_bottomrnn_from{}_to{}_seed{}_best_auprc.pt'.format(args.few_shot, args.source_file, args.test_file, args.seed)
        self.target_best_path = os.path.join(args.path, args.item, model_directory, args.test_file, file_target_name, target_filename)
        print('Save path: {}'.format(self.target_best_path))

        # load parameters
        ckpt = torch.load(source_best_path)
        pretrained_dict = ckpt['model_state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if not 'embedding' in k}
        self.model.load_state_dict(pretrained_dict, strict=False)
        print('Model partially loaded!')

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.early_stopping = EarlyStopping(patience=20, verbose=True)


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
                item_id, item_target, seq_len, word_seq_len = sample
                item_id = item_id.to(self.device)
                item_target = item_target.to(self.device)

                y_pred = self.model(item_id, seq_len, word_seq_len)
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
                            'epochs': n_epoch}, self.target_best_path)
                print('Model parameter saved at epoch {}'.format(n_epoch))

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
                print('Early stopping!')
                break

        self.test()


    def evaluation(self):
        self.model.eval()
        preds_eval = []
        truths_eval = []
        avg_eval_loss = 0.

        with torch.no_grad():
            for iter, sample in enumerate(self.eval_dataloader):
                item_id, item_target, seq_len, word_seq_len = sample
                item_id = item_id.to(self.device)
                item_target = item_target.to(self.device)

                y_pred = self.model(item_id, seq_len, word_seq_len)
                if self.BCE and self.target != 'dx_depth1_unique':
                    loss = self.criterion(y_pred, item_target.unsqueeze(1).float().to(self.device))
                else:
                    loss = self.criterion(y_pred, item_target.float().to(self.device))
                avg_eval_loss += loss.item() / len(self.eval_dataloader)

                probs_eval = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_eval += list(probs_eval.flatten())
                truths_eval += list(item_target.detach().cpu().numpy().flatten())

            auroc_eval = roc_auc_score(truths_eval, preds_eval)
            auprc_eval = average_precision_score(truths_eval, preds_eval, average='micro')

        return avg_eval_loss, auroc_eval, auprc_eval

    def test(self):
        ckpt = torch.load(self.target_best_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        preds_test = []
        truths_test = []
        avg_test_loss = 0.

        with torch.no_grad():
            for iter, sample in enumerate(self.test_dataloader):
                item_id, item_target, seq_len, word_seq_len = sample
                item_id = item_id.to(self.device)
                item_target = item_target.to(self.device)

                y_pred = self.model(item_id, seq_len, word_seq_len)
                if self.BCE and self.target != 'dx_depth1_unique':
                    loss = self.criterion(y_pred, item_target.unsqueeze(1).float().to(self.device))
                else:
                    loss = self.criterion(y_pred, item_target.float().to(self.device))
                avg_test_loss += loss.item() / len(self.test_dataloader)

                probs_eval = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_test += list(probs_eval.flatten())
                truths_test += list(item_target.detach().cpu().numpy().flatten())

            auroc_test = roc_auc_score(truths_test, preds_test)
            auprc_test = average_precision_score(truths_test, preds_test, average='micro')

            wandb.log({'test_loss': avg_test_loss,
                       'test_auroc': auroc_test,
                       'test_auprc': auprc_test})

            print('[Test]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_test_loss, auroc_test, auprc_test))

    def zero_test(self):
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

            wandb.log({'test_loss': avg_test_loss,
                           'test_auroc': auroc_test,
                           'test_auprc': auprc_test})

            print('[Test]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_test_loss, auroc_test, auprc_test))






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_induced', action='store_true')
    parser.add_argument('--source_file', choices=['mimic', 'eicu', 'both'], type=str, default='mimic')
    parser.add_argument('--test_file', choices=['mimic', 'eicu'], type=str)
    parser.add_argument('--target', choices=['readmission', 'mortality', 'los>3day', 'los>7day', 'dx_depth1_unique'], type=str, default='readmission')
    parser.add_argument('--item', choices=['lab', 'med', 'inf', 'all'], type=str, default='lab')
    parser.add_argument('--few_shot', choices=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], type=float)
    parser.add_argument('--time_window', choices=['12', '24', '36', '48', 'Total'], type=str, default='12')
    parser.add_argument('--rnn_model_type', choices=['gru', 'lstm'], type=str, default='gru')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--bottom_embedding_size', type=int)
    parser.add_argument('--bottom_hidden_size', type=int)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_length', type=str, default='150')
    parser.add_argument('--bert_model', choices=['bert', 'bio_clinical_bert', 'bio_bert', 'pubmed_bert', 'blue_bert', 'bert_mini', 'bert_tiny', 'bert_small'], type=str)
    parser.add_argument('--bert_freeze', action='store_true')
    parser.add_argument('--cls_freeze', action='store_true')
    parser.add_argument('--input_path', type=str, default='/home/jylee/data/pretrained_ehr/input_data/')
    parser.add_argument('--path', type=str, default='/home/jylee/data/pretrained_ehr/output/KDD_output2/')
    parser.add_argument('--device_number', type=str)
    parser.add_argument('--notes', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--concat', action='store_true')
    parser.add_argument('--only_BCE', action='store_true')
    parser.add_argument('--transformer', action='store_true')
    parser.add_argument('--transformer_layers', type=int, default=2)
    parser.add_argument('--transformer_attn_heads', type=int, default=8)
    parser.add_argument('--transformer_hidden_dim', type=int, default=256)
    parser.add_argument('--wandb_project_name', type=str)
    args = parser.parse_args()

    args.word_max_length = 30 if args.test_file == 'mimic' else 36

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_number)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # hyperparameter tuning
    args.dropout = 0.3
    args.embedding_dim = 128
    args.hidden_dim = 256

    mp.set_sharing_strategy('file_system')

    print('start running')
    SEED = [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029]

    for seed in SEED:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True

        args.seed = seed

        trainer = RNNBottomTrainer(args, device)

        if args.few_shot == 0.0:
            trainer.zero_test()
        else:
            trainer.train()

if __name__ == '__main__':
    main()
