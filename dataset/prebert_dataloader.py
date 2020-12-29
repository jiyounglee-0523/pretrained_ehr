import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer
import numpy as np

import pickle

"""
TO-DO: Implement Total (eicu and mimic combined)
"""

def bertinduced_get_dataloader(args, validation_index, data_type='train'):
    if data_type == 'train':
        train_data = healthcare_dataset(args, validation_index, data_type)
        dataloader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)

    elif data_type == 'eval':
        eval_data = healthcare_dataset(args, validation_index, data_type)
        dataloader = DataLoader(dataset=eval_data, batch_size=args.batch_size, shuffle=True)

    return dataloader

class healthcare_dataset(Dataset):
    def __init__(self, args, validation_index, data_type):
        source_file = args.source_file
        self.target = args.target
        item = args.item
        self.max_length = args.max_length
        time_window = args.time_window
        self.word_max_length = args.word_max_length

        if source_file == 'both':
            path = '/home/jylee/data/pretrained_ehr/mimic_{}_{}_{}.pkl'.format(time_window, item, self.max_length)
            mimic = pickle.load(open(path,'rb'))

            path = '/home/jylee/data/pretrained_ehr/eicu_{}_{}_{}.pkl'.format(time_window, item, self.max_length)
            eicu = pickle.load(open(path, 'rb'))

            mimic = mimic.rename({'HADM_ID': 'ID'}, axis='columns')
            eicu = eicu.rename({'patientunitstayid': 'ID'}, axis='columns')

            mimic_item_name, mimic_item_offset, mimic_item_offset_order, mimic_item_target = self.preprocess(mimic, validation_index, data_type, item, time_window, self.target)
            eicu_item_name, eicu_item_offset, eicu_item_offset_order, eicu_item_target = self.preprocess(eicu, validation_index, data_type, item, time_window, self.target)

            mimic_item_name.extend(eicu_item_name)
            self.item_name = mimic_item_name
            self.item_offset = torch.cat((mimic_item_offset, eicu_item_offset), dim=0)
            self.item_offset_order = torch.cat((mimic_item_offset_order, eicu_item_offset_order), dim=0)
            if self.target == 'dx_depth1_unique':
                mimic_item_target.extend(eicu_item_target)
                self.item_target = mimic_item_target
            else:
                self.item_target = torch.cat((mimic_item_target, eicu_item_target))

        else:
            path = '/home/jylee/data/pretrained_ehr/{}_{}_{}_{}.pkl'.format(source_file, time_window, item, self.max_length)
            data = pickle.load(open(path, 'rb'))

            # change column name
            if source_file == 'mimic':
                data = data.rename({'HADM_ID': 'ID'}, axis='columns')

            elif source_file == 'eicu':
                data = data.rename({'patientunitstayid': 'ID'}, axis='columns')

            self.item_name, self.item_offset, self.item_offset_order, self.item_target = self.preprocess(data, validation_index, data_type, item, time_window, self.target)

        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    def __len__(self):
        return self.item_offset.size(0)

    def __getitem__(self, item):
        single_item_name = self.item_name[item]
        seq_len = torch.Tensor([len(single_item_name)])
        # pad
        pad_length = int(self.max_length) - len(single_item_name)
        single_item_name.extend(['[PAD]'] * pad_length)
        assert len(single_item_name) == int(self.max_length), "item_name padded wrong"

        single_item_name = self.tokenizer(single_item_name, padding = 'max_length', return_tensors='pt', max_length=self.word_max_length)  # seq_len x words


        # single_item_offset = torch.cat([self.item_offset[item]] * seq_len, dim=0)
        # single_item_offset_order = torch.cat([self.item_offset_order[item]] * seq_len, dim=0)
        # single_item_offset = self.item_offset[item]
        # single_item_offset_order = self.item_offset_order[item]
        single_target = self.item_target[item]
        if self.target == 'dx_depth1_unique':
            single_target = [int(j) for j in single_target]
            target_list = torch.Tensor(single_target).long()

            single_target = torch.zeros(18)
            single_target[target_list - 1] = 1     # shape of 18

        # implement single_length later

        return single_item_name, single_target, seq_len

    def preprocess(self, cohort, validation_index, data_type, item, time_window, target):
        if time_window == 'Total':
            raise NotImplementedError

        else:
            name_window = '{}_name_{}hr'.format(item, time_window)
            offset_window = 'order_offset_{}hr'.format(time_window)
            offset_order_window = '{}_offset_order_{}hr'.format(item, time_window)
            target_fold = '{}_fold'.format(target)
            if target == 'dx_depth1_unique':
                target_fold = 'dx_fold'

        # extract cohort
        cohort = cohort[['ID', name_window, offset_window, offset_order_window, target, target_fold]]
        cohort = cohort[cohort[target_fold] != 0]   # 0 is for test dataset

        if data_type == 'train':
            cohort = cohort[cohort[target_fold] != validation_index]
        elif data_type == 'eval':
            cohort = cohort[cohort[target_fold] == validation_index]

        # pad
        item_name = cohort[name_window].values.tolist()    # list of item_names

        item_offset_order = cohort[offset_order_window].apply(lambda x: torch.Tensor(x)).values.tolist()
        item_offset_order = pad_sequence(item_offset_order, batch_first=True)  # shape of (B, max_len)

        item_offset = cohort[offset_window].apply(lambda x: torch.Tensor(x)).values.tolist()
        item_offset = pad_sequence(item_offset, batch_first=True)

        # target
        if target == 'dx_depth1_unique':
            item_target = cohort[target].values.tolist()
        else:
            item_target = torch.LongTensor(cohort[target].values.tolist())  # shape of (B)

        return item_name, item_offset, item_offset_order, item_target
