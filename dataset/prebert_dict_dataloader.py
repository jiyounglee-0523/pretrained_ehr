import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pickle
import os
import re

from dataset.prebert_dataloader import healthcare_dataset


def bertinduced_dict_get_dataloader(args, validation_index, data_type='train'):
    if data_type == 'train':
        train_data = bert_dict_dataset(args, validation_index, data_type)
        dataloader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)

    elif data_type == 'eval':
        eval_data = bert_dict_dataset(args, validation_index, data_type)
        dataloader = DataLoader(dataset=eval_data, batch_size=args.batch_size, shuffle=True)

    return dataloader



class bert_dict_dataset(healthcare_dataset):
    def __init__(self, args, validation_index, data_type):
        super(bert_dict_dataset, self).__init__(args, validation_index, data_type)

        del self.tokenizer

        if args.source_file == 'both':
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
        for name in single_item_name:
            name = re.sub(r'[.,/|!-?"\':;~()\[\]]', '', name)
            single_embedding = self.word2embed[name]
            embedding.append(single_embedding)

        embedding = torch.Tensor(np.stack(embedding, axis=0))    # shape of (seq_len, 768)
        # padding
        padding = torch.zeros(int(self.max_length) - embedding.size(0), embedding.size(1))
        embedding = torch.cat((embedding, padding), dim=0)    # shape of (max_length, 768)
        assert list(embedding.shape)[0] == int(self.max_length), "padding wrong!"

        single_target = self.item_target[item]
        if self.target == 'dx_depth1_unique':
            single_target = [int(j) for j in single_target]
            target_list = torch.Tensor(single_target).long()

            single_target = torch.zeros(18)
            single_target[target_list - 1] = 1   # shape of 18

        # implement single_length later

        return embedding, single_target, seq_len