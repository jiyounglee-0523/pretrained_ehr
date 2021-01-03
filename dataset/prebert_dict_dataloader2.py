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
            path = '/home/jylee/data/pretrained_ehr/input_data/mimic_{}_{}_{}_{}.pkl'.format(time_window, item, self.max_length, args.seed)
            mimic = pickle.load(open(path, 'rb'))

            path = '/home/jylee/data/pretrained_ehr/input_data/eicu_{}_{}_{}_{}.pkl'.format(time_window, item, self.max_length, args.seed)
            eicu = pickle.load(open(path, 'rb'))

            mimic = mimic.rename({'HADM_ID': 'ID'}, axis='columns')
            eicu = eicu.rename({'patientunitstayid': 'ID'}, axis='columns')

            mimic_vocab_path = os.path.join('/home/jylee/data/pretrained_ehr', 'mimic_{}_word2embed.pkl'.format(args.item))
            mimic_word2embed = pickle.load(open(mimic_vocab_path, 'rb'))

            eicu_vocab_path = os.path.join('/home/jylee/data/pretrained_ehr', 'eicu_{}_word2embed.pkl'.format(args.item))
            eicu_word2embed = pickle.load(open(eicu_vocab_path, 'rb'))

            mimic_word2embed.update(eicu_word2embed)
            self.word2embed = mimic_word2embed

        else:
            vocab_path = os.path.join('/home/jylee/data/pretrained_ehr', '{}_{}_word2embed.pkl'.format(args.source_file, args.item))
            self.word2embed = pickle.load(open(vocab_path, 'rb'))

    def __len__(self):
        return self.item_offset.size(0)

    def __getitem__(self, item):
        single_item_name = self.item_name[item]
        seq_len = torch.Tensor([len(single_item_name)])
        embedding = []

        single_item_name = [re.sub(r'[.,/|!-?"\':;~()\[\]]', '', i) for i in single_item_name]

        def embed_dict(x):
            return self.word2embed[x]
        embedding = list(map(embed_dict, single_item_name))     # list with length seq_len
        embedding = torch.Tensor(np.stack(embedding, axis=0))   # tensor of shape (seq_len, 768)

        padding = torch.zeros(int(self.max_length) - embedding.size(0), embedding.size(1))
        embedding = torch.cat((embedding, padding), dim=0)
        assert list(embedding.shape)[0] == int(self.max_length), "padding wrong!"

        single_target = self.item_target[item]
        if self.target == 'dx_depth1_unique':
            single_target = [int(j) for j in single_target]
            target_list = torch.Tensor(single_target).long()

            single_target = torch.zeros(18)
            single_target[target_list - 1] = 1   # shape of 18

        # implement single_length later

        return embedding, single_target, seq_len

    def preprocess(self, ):