import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
import wandb
import os

from models.rnn_bert_dict import dict_post_RNN
from utils.loss import *
from utils.trainer_utils import *
from trainer.bert_induced_trainer import Bert_Trainer

class bert_dict_Trainer(Bert_Trainer):
    def __init__(self, args, train_dataloader, valid_dataloader, device, valid_index):
        super(bert_dict_Trainer, self).__init__(args, train_dataloader, valid_dataloader, device, valid_index)

        if args.target == 'dx_depth1_unique':
            output_size = 18
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            output_size = 1
            self.criterion = FocalLoss()

        self.model = dict_post_RNN(args, output_size).to(self.device)
