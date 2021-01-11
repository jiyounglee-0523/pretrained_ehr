import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pickle
import os
import re

from dataset.prebert_dataloader import healthcare_dataset


def bertinduced_dict_get_dataloader(args, data_type='train'):
    if data_type == 'train':
        train_data = bert_dict_dataset(args, data_type)
        dataloader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=32)

    elif data_type == 'eval':
        eval_data = bert_dict_dataset(args, data_type)
        dataloader = DataLoader(dataset=eval_data, batch_size=args.batch_size, shuffle=True, num_workers=32)

    elif data_type == 'test':
        test_data = bert_dict_dataset(args, data_type)
        dataloader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, num_workers=32)

    return dataloader



class bert_dict_dataset(Dataset):
    def __init__(self, args, data_type):
        source_file = args.source_file
        self.target = args.target
        item = args.item
        self.max_length = args.max_length
        time_window = args.time_window
        self.word_max_length = args.word_max_length

        if args.source_file == 'both':
            path = os.path.join('/home/jylee/data/pretrained_ehr/input_data', item,
                                'mimic_{}_{}_{}_{}.pkl'.format(source_file, time_window, item, self.max_length, args.seed))
            mimic = pickle.load(open(path, 'rb'))

            path = os.path.join('/home/jylee/data/pretrained_ehr/input_data', item,
                                'eicu_{}_{}_{}_{}.pkl'.format(source_file, time_window, item, self.max_length, args.seed))
            eicu = pickle.load(open(path, 'rb'))

            mimic = mimic.rename({'HADM_ID': 'ID'}, axis='columns')
            eicu = eicu.rename({'patientunitstayid': 'ID'}, axis='columns')

            mimic_item_name, mimic_item_target = self.preprocess(mimic, data_type, item, time_window, self.target)
            eicu_item_name, eicu_item_target = self.preprocess(eicu, data_type, item, time_window, self.target)

            mimic_item_name.extend(eicu_item_name)
            self.item_name = mimic_item_name
            if self.target == 'dx_depth1_unique':
                mimic_item_target.extend(eicu_item_target)
                self.item_target = mimic_item_target
            else:
                self.item_target = torch.cat((mimic_item_target, eicu_item_target))


            mimic_vocab_path = os.path.join('/home/jylee/data/pretrained_ehr', 'mimic_{}_word2embed.pkl'.format(args.item))
            mimic_word2embed = pickle.load(open(mimic_vocab_path, 'rb'))

            eicu_vocab_path = os.path.join('/home/jylee/data/pretrained_ehr', 'eicu_{}_word2embed.pkl'.format(args.item))
            eicu_word2embed = pickle.load(open(eicu_vocab_path, 'rb'))

            mimic_word2embed.update(eicu_word2embed)
            self.word2embed = mimic_word2embed

        else:
            path = os.path.join('/home/jylee/data/pretrained_ehr/input_data', item, '{}_{}_{}_{}_{}.pkl'.format(source_file, time_window, item, self.max_length, args.seed))
            data = pickle.load(open(path, 'rb'))

            if source_file == 'mimic':
                data = data.rename({'HADM_ID': 'ID'}, axis='columns')

            elif source_file == 'eicu':
                data = data.rename({'patientunitstayid': 'ID'}, axis='columns')

            self.item_name, self.item_target = self.preprocess(data, data_type, item, time_window, self.target)

            # check data path
            vocab_path = os.path.join('/home/jylee/data/pretrained_ehr/input_data/embed_vocab_file', item,
                                      '{}_{}_{}_{}_word2embed.pkl'.format(source_file, item, time_window, args.bert_model))
            self.id_dict = pickle.load(open(vocab_path, 'rb'))

    def __len__(self):
        return len(self.item_name)

    def __getitem__(self, item):
        single_item_name = self.item_name[item]
        seq_len = torch.Tensor([len(single_item_name)])
        embedding = []

        def embed_dict(x):
            return self.id_dict[x]
        embedding = list(map(embed_dict, single_item_name))     # list with length seq_len
        embedding = torch.Tensor(embedding)

        padding = torch.zeros(int(self.max_length) - embedding.size(0))
        embedding = torch.cat((embedding, padding), dim=-1)
        assert list(embedding.shape)[0] == int(self.max_length), "padding wrong!"

        single_target = self.item_target[item]
        if self.target == 'dx_depth1_unique':
            single_target = [int(j) for j in single_target]
            target_list = torch.Tensor(single_target).long()

            single_target = torch.zeros(18)
            single_target[target_list - 1] = 1   # shape of 18

        # implement single_length later

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
        item_name = cohort[name_window].values.tolist()

        ## offset order? offset?

        if target == 'dx_depth1_unique':
            item_target = cohort[target].values.tolist()
        else:
            item_target = torch.LongTensor(cohort[target].values.tolist())  # shape of B

        return item_name, item_target