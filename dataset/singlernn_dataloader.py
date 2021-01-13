import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import os

import pickle

def singlernn_get_dataloader(args, data_type = 'train'):
    if data_type == 'train':
        train_data = eicu_dataset(args, data_type)
        dataloader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)

    elif data_type == 'eval':
        eval_data = eicu_dataset(args, data_type)
        dataloader = DataLoader(dataset=eval_data, batch_size=args.batch_size, shuffle=True)

    elif data_type == 'test':
        test_data =  eicu_dataset(args, data_type)
        dataloader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    return dataloader


class eicu_dataset(Dataset):
    def __init__(self, args, data_type):
        source_file = args.source_file
        self.target = args.target
        item = args.item
        max_length = args.max_length
        time_window = args.time_window

        if args.concat:
            path = os.path.join('/home/jylee/data/pretrained_ehr/input_data', item,
                            '{}_{}_{}_{}_{}_concat.pkl'.format(source_file, time_window, item, max_length, args.seed))
        elif not args.concat:
            path = os.path.join('/home/jylee/data/pretrained_ehr/input_data', item,
                            '{}_{}_{}_{}_{}.pkl'.format(source_file, time_window, item, max_length, args.seed))
        data = pickle.load(open(path, 'rb'))

        # change column name
        if source_file == 'mimic':
            data = data.rename({'HADM_ID':'ID'}, axis='columns')
        elif source_file == 'eicu':
            data = data.rename({'patientunitstayid':'ID'}, axis='columns')
        else:
            raise NotImplementedError

        self.item_id, item_offset, item_offset_order, self.item_target = self.preprocess(data, data_type, item, time_window, self.target)


    def __len__(self):
        return self.item_id.size(0)

    def __getitem__(self, item):
        single_item_id = self.item_id[item]
        # single_item_offset = self.item_offset[item]
        # single_item_offset_order = self.item_offset_order[item]
        single_target = self.item_target[item]
        single_length = torch.LongTensor([torch.max(torch.nonzero(single_item_id.data)) + 1])

        if self.target == 'dx_depth1_unique':
            single_target = [int(j) for j in single_target]
            target_list = torch.Tensor(single_target).long()

            single_target = torch.zeros(18)
            single_target[target_list - 1] = 1  # shape of 18

        return single_item_id, single_target, single_length


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
        cohort = cohort[['ID', id_window, offset_window, offset_order_window, target, target_fold]]

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

        item_offset_order = cohort[offset_order_window].apply(lambda x: torch.Tensor(x)).values.tolist()
        item_offset_order = pad_sequence(item_offset_order, batch_first=True)  # shape of (B, max_len)

        item_offset = cohort[offset_window].apply(lambda x: torch.Tensor(x)).values.tolist()
        item_offset = pad_sequence(item_offset, batch_first=True)

        # target
        if target == 'dx_depth1_unique':
            item_target = cohort[target].values.tolist()
        else:
            item_target = torch.LongTensor(cohort[target].values.tolist())     # shape of (B)

        return item_id, item_offset, item_offset_order, item_target



