import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import numpy as np

import pickle5 as pickle

def get_dataloader(args, data_type = 'train'):
    if data_type == 'train':
        train_data = eicu_dataset(args)
        dataloader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)

    elif data_type == 'eval':
        eval_data = eicu_dataset(args)
        dataloader = DataLoader(dataset=eval_data, batch_size=args.batch_size, shuffle=True)

    return dataloader


class eicu_dataset(Dataset):
    def __init__(self, args, offset=False, down_sampling=False, padding=True):
        source_file = args.source_file
        target = args.target
        item = args.item
        max_length = args.max_length
        time_window = args.time_window

        path = './{}_{}_{}_{}.pkl'.format(source_file, time_window, item, max_length)
        data = pickle.load(open(path, 'rb'))

        # change column name
        if source_file == 'mimic':
            data = data.rename({'HADM_ID':'ID'}, axis='columns')
        elif source_file == 'eicu':
            data = data.rename({'patientunitstayid':'ID'}, axis='columns')
        else:
            raise NotImplementedError

        self.item_id, self.item_offset, self.item_offset_order, self.item_target = self.preprocess(data, item, time_window, target, offset, down_sampling, padding)


    def __len__(self):
        return self.item_id.size(0)

    def __getitem__(self, item):
        single_item_id = self.item_id[item]
        single_item_offset = self.item_offset[item]
        single_item_offset_order = self.item_offset_order[item]
        single_target = self.item_target[item]
        single_length = torch.LongTensor([torch.max(torch.nonzero(single_item_id.data)) + 1])

        return single_item_id, single_item_offset, single_item_offset_order, single_length, single_target


    def preprocess(self, cohort, item, time_window, target, offset, down_sampling, padding):
        # time window
        if time_window == 'Total':
            name_window = '{}_name'.format(item)
            offset_window = '{}_offset'.format(item)
            offset_order_window = '{}_offset_order'.format(item)     ##### 바꿔야 한다!!
            id_window = None   ### 바꿔야 한다!!!
        else:
            name_window = '{}_name_{}hr'.format(item, time_window)
            offset_window = '{}_offset_{}hr'.format(item, time_window)
            offset_order_window = 'offset_order_{}hr'.format(time_window)               ##### 바꿔야 한다!! '{}_offset_order_{}jr'.format(item, time_window
            id_window = '{}_id_{}hr'.format(item, time_window)

        # extract cohort
        cohort = cohort[['ID', id_window, offset_window, offset_order_window, target]]

        # pad
        if padding:
            item_id = cohort[id_window].apply(lambda x: torch.Tensor(x)).values.tolist()
            item_id = pad_sequence(item_id, batch_first=True)   # shape of (B, max_len)

            item_offset_order = cohort[offset_order_window].apply(lambda x: torch.Tensor(x)).values.tolist()
            item_offset_order = pad_sequence(item_offset_order, batch_first=True)  # shape of (B, max_len)

            item_offset = cohort[offset_window].apply(lambda x: torch.Tensor(x)).values.tolist()
            item_offset = pad_sequence(item_offset, batch_first=True)

        # target
        item_target = torch.LongTensor(cohort[target].values.tolist())     # shape of (B)

        return item_id, item_offset, item_offset_order, item_target



