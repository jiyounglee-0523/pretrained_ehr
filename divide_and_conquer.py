import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import roc_auc_score, average_precision_score

import pickle
import os
import argparse
import easydict


def CodeEmb_get_dataloader(args, data_type = 'train', data_name = None):
    if data_type == 'train':
        train_data = CodeEmb_dataset(args, data_type, data_name=data_name)
        dataloader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)

    elif data_type == 'eval':
        eval_data = CodeEmb_dataset(args, data_type, data_name=data_name)
        dataloader = DataLoader(dataset=eval_data, batch_size=args.batch_size, shuffle=True)

    elif data_type == 'test':
        test_data = CodeEmb_dataset(args, data_type, data_name=data_name)
        dataloader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    return dataloader


class CodeEmb_dataset(Dataset):
    def __init__(self, args, data_type, data_name=None):
        if data_name is None:
            source_file = args.source_file
        else:
            source_file = data_name
        self.source_file = source_file
        self.target = args.target
        item = args.item
        max_length = args.max_length
        self.max_length = max_length
        time_window = args.time_window

        if source_file == 'both':


            mimic_path = os.path.join(args.input_path[:-1], item,
                                'mimic_{}_{}_{}_{}.pkl'.format(time_window, item, max_length, args.seed))
            eicu_path = os.path.join(args.input_path[:-1], item,
                                      'eicu_{}_{}_{}_{}.pkl'.format(time_window, item, max_length, args.seed))
            mimic = pickle.load(open(mimic_path, 'rb'))
            eicu = pickle.load(open(eicu_path, 'rb'))

            mimic = mimic.rename({'HADM_ID': 'ID'}, axis='columns')
            eicu = eicu.rename({'patientunitstayid': 'ID'}, axis='columns')

            mimic_item_name, mimic_item_target, mimic_item_offset_order = self.preprocess(mimic, data_type, item, time_window, self.target)
            eicu_item_name, eicu_item_target, eicu_item_offset_order = self.preprocess(eicu, data_type, item, time_window, self.target)

            mimic_item_name.extend(eicu_item_name)
            self.item_name = mimic_item_name

            mimic_item_offset_order = list(mimic_item_offset_order)
            eicu_item_offset_order = list(eicu_item_offset_order)
            mimic_item_offset_order.extend(eicu_item_offset_order)
            self.item_offset_order = pad_sequence(mimic_item_offset_order, batch_first=True)

            if self.target == 'dx_depth1_unique':
                mimic_item_target.extend(eicu_item_target)
                self.item_target = mimic_item_target
            else:
                self.item_target = torch.cat((mimic_item_target, eicu_item_target))

        else:
            path = os.path.join(args.input_path[:-1], item,
                            '{}_{}_{}_{}_{}.pkl'.format(source_file, time_window, item, max_length, args.seed))
            data = pickle.load(open(path, 'rb'))

            # change column name
            if source_file == 'mimic':
                data = data.rename({'HADM_ID':'ID'}, axis='columns')
            elif source_file == 'eicu':
                data = data.rename({'patientunitstayid':'ID'}, axis='columns')

            self.item_name, self.item_target, self.item_offset_order = self.preprocess(data, data_type, item, time_window, self.target)


        vocab_path = os.path.join(args.input_path + 'embed_vocab_file', item,
                                      '{}_{}_{}_{}_word2embed.pkl'.format(args.source_file, item, time_window,
                                                                          args.bert_model))
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
        embedding = list(map(embed_dict, single_item_name))
        embedding = torch.Tensor(embedding)

        padding = torch.zeros(int(self.max_length) - embedding.size(0))
        embedding = torch.cat((embedding, padding), dim=-1)

        return embedding, single_target, seq_len


    def preprocess(self, cohort, data_type, item, time_window, target):
        # time window
        if time_window == 'Total':
            name_window = '{}_name'.format(item)
            offset_window = 'order_offset'.format(item)
            offset_order_window = '{}_offset_order'.format(item)     ##### 바꿔야 한다!!
            id_window = '{}_id_{}hr'.format(item, time_window)
            target_fold = '{}_fold'.format(target)
        else:
            name_window = '{}_name_{}hr'.format(item, time_window)
            offset_window = 'order_offset_{}hr'.format(time_window)    ## we should reanme with using item
            offset_order_window = '{}_offset_order_{}hr'.format(item, time_window)
            id_window = '{}_id_{}hr'.format(item, time_window)
            target_fold = '{}_fold'.format(target)
            if target == 'dx_depth1_unique':
                target_fold = 'dx_fold'

        # extract cohort
        cohort = cohort[['ID', name_window, id_window, offset_window, offset_order_window, target, target_fold]]

        if data_type == 'train':
            cohort = cohort[cohort[target_fold] == 1]
        elif data_type == 'eval':
            cohort = cohort[cohort[target_fold] == 2]
        elif data_type == 'test':
            cohort = cohort[cohort[target_fold] == 0]

        # drop with null item
        cohort = cohort[cohort.astype(str)[id_window] != '[]']

        # pad
        item_id = cohort[id_window].apply(lambda x: torch.Tensor(x)).values.tolist()
        item_id = pad_sequence(item_id, batch_first=True)   # shape of (B, max_len)

        item_name = cohort[name_window].values.tolist()

        item_offset_order = cohort[offset_order_window].apply(lambda x: torch.Tensor(x)).values.tolist()
        item_offset_order = pad_sequence(item_offset_order, batch_first=True)  # shape of (B, max_len)

        # target
        if target == 'dx_depth1_unique':
            item_target = cohort[target].values.tolist()
        else:
            item_target = torch.LongTensor(cohort[target].values.tolist())     # shape of (B)

        return item_name, item_target, item_offset_order

class CodeEmb(nn.Module):
    def __init__(self, args, vocab_size, output_size, device, n_layers = 1):
        super(CodeEmb, self).__init__()
        embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        dropout = args.dropout

        self.device = device

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.model = nn.GRU(embedding_dim, self.hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.output_fc = nn.Linear(self.hidden_dim, output_size)

    def forward(self, x, lengths):
        x = self.embedding(x.long().to(self.device))

        lengths = lengths.squeeze(-1).long()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.model(packed)
        output_seq, output_len = pad_packed_sequence(output, batch_first=True)

        i = range(x.size(0))

        output = output_seq[i, lengths -1, :]
        output = self.output_fc(output)
        return output




def main():
    target_list = ['readmission', 'mortality', 'los>3day', 'los>7day', 'dx_depth1_unique']
    seed_list = [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029]

    device = torch.device('cuda:0')

    for target in target_list:
        for seed in seed_list:
            mimic_args = easydict.EasyDict({
                "DescEmb": False,
                "source_file": 'mimic',
                "target": target,
                "item": 'all',
                "time_window": '12',
                "batch_size": 512,
                "n_epochs": 1000,
                "lr": 1e-4,
                "max_length": 150,
                "bert_model": 'bio_clinical_bert',
                "cls_freeze": False,
                "input_path": '/home/jylee/data/pretrained_ehr/input_data_old/',
                "path": '/home/jylee/data/pretrained_ehr/output/KDD_output/',
                "device_number": 0,
                "dropout": 0.3,
                "embedding_dim": 128,
                "hidden_dim": 256,
                "seed": seed})

            eicu_args = easydict.EasyDict({
                "DescEmb": False,
                "source_file": 'eicu',
                "target": target,
                "item": 'all',
                "time_window": '12',
                "batch_size": 512,
                "n_epochs": 1000,
                "lr": 1e-4,
                "max_length": 150,
                "bert_model": 'bio_clinical_bert',
                "cls_freeze": False,
                "input_path": '/home/jylee/data/pretrained_ehr/input_data_old/',
                "path": '/home/jylee/data/pretrained_ehr/output/KDD_output/',
                "device_number": 0,
                "dropout": 0.3,
                "embedding_dim": 128,
                "hidden_dim": 256,
                "seed": seed})

            mimic_model_name = 'trained_single_rnn_{}_best_auprc.pt'.format(mimic_args.seed)
            eicu_model_name = 'trained_single_rnn_{}_best_auprc.pt'.format(eicu_args.seed)

            target_directory = target
            if target_directory == 'los>3day':
                target_directory = 'los_3days'
            elif target_directory == 'los>7day':
                target_directory = 'los_7days'

            mimic_path = '/home/jylee/data/pretrained_ehr/output/KDD_output/all/singleRNN/mimic/{}'.format(target_directory)
            eicu_path = '/home/jylee/data/pretrained_ehr/output/KDD_output/all/singleRNN/eicu/{}'.format(target_directory)

            mimic_path = os.path.join(mimic_path, mimic_model_name)
            eicu_path = os.path.join(eicu_path, eicu_model_name)

            if target == 'dx_depth1_unique':
                output_size = 18
            else:
                output_size = 1

            mimic_model = CodeEmb(mimic_args, vocab_size=2377, output_size=output_size, device=device).to(device)
            eicu_model = CodeEmb(eicu_args, vocab_size=1344, output_size=output_size, device=device).to(device)

            mimic_param = torch.load(mimic_path)
            eicu_param = torch.load(eicu_path)

            mimic_model.load_state_dict(mimic_param['model_state_dict'])
            eicu_model.load_state_dict(eicu_param['model_state_dict'])
            print('parameters are successfully loaded! {}{}'.format(target, seed))

            mimic_model.eval()
            eicu_model.eval()

            mimic_test_dataloader = CodeEmb_get_dataloader(mimic_args, data_type='test')
            eicu_test_dataloader = CodeEmb_get_dataloader(eicu_args, data_type='test')

            mimic_model.eval()
            eicu_model.eval()

            mimic_preds_test = []
            mimic_truths_test = []
            mimic_avg_test_loss = 0.
            eicu_preds_test = []
            eicu_truths_test = []
            eicu_avg_test_loss = 0.

            with torch.no_grad():
                for iter, sample in enumerate(mimic_test_dataloader):
                    item_embed, item_target, seq_len = sample
                    item_embed = item_embed.to(device)
                    item_target = item_target.to(device)
                    y_pred = mimic_model(item_embed, seq_len)

                    mimic_probs_test = torch.sigmoid(y_pred).detach().cpu().numpy()
                    mimic_preds_test += list(mimic_probs_test.flatten())
                    mimic_truths_test += list(item_target.detach().cpu().numpy().flatten())

                for iter, sample in enumerate(eicu_test_dataloader):
                    item_embed, item_target, seq_len = sample
                    item_embed = item_embed.to(device)
                    item_target = item_target.to(device)
                    y_pred = eicu_model(item_embed, seq_len)

                    eicu_probs_test = torch.sigmoid(y_pred).detach().cpu().numpy()
                    eicu_preds_test += list(eicu_probs_test.flatten())
                    eicu_truths_test += list(item_target.detach().cpu().numpy().flatten())

                truths_test = mimic_truths_test + eicu_truths_test
                preds_test = mimic_preds_test + eicu_preds_test

                test_auprc = average_precision_score(truths_test, preds_test, average='micro')
                print('[{}-{}], test_auprc:{:.3f}'.format(target, seed, test_auprc))


if __name__ == '__main__':
    main()