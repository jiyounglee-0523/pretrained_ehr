import torch
import torch.nn as nn

import wandb
import os

from dataset.prebert_dict_dataloader import bertinduced_dict_get_dataloader
from models.transformer import Transformer
from models.rnn_models import RNNmodels
from models.rnn_bert_dict import dict_post_RNN
from utils.trainer_utils import EarlyStopping


class FinetuningBoth():
    def __init__(self, args, target_file, before_best_eval_path, device):
        assert args.source_file == 'both', 'finetuning should be conducted at both'
        assert target_file == 'mimic' or target_file=='eicu', 'target file should be either mimic or eicu'
        self.dataloader = bertinduced_dict_get_dataloader(args, data_type='train', data_name=target_file)
        self.valid_dataloader = bertinduced_dict_get_dataloader(args, data_type='eval', data_name=target_file)
        self.test_dataloader = bertinduced_dict_get_dataloader(args, data_type='test', data_name=target_file)

        self.device = device
        self.debug = args.debug
        self.BCE = args.only_BCE
        self.target = args.target
        self.source_file = args.source_file    # both
        self.lr_scheduler = args.lr_scheduler

        if not self.debug:
            wandb.init(project=args.wandb_project_name, entity='pretrained_ehr', config=args, reinit=True)

        lr = args.lr
        self.lr = lr
        self.n_epochs = args.n_epochs

        file_target_name = args.target
        if file_target_name == 'los>3day':
            file_target_name = 'los_3days'
        elif file_target_name == 'los>7day':
            file_target_name = 'los_7days'

        if not args.transformer:
            raise NotImplementedError

        elif args.transformer:
            assert not args.cls_freeze and not args.concat and args.only_BCE, 'no cls_freeze, no concat, only BCE'
            filename = 'finetune_transformer_{}_{}_{}_{}_{}_onlyBCE'.format(target_file, args.bert_model, args.seed, args.lr_scheduler, args.lr)   # adjust this

        path = os.path.join(args.path, args.item, 'cls_learnable', args.source_file, file_target_name, filename)
        print('Model will be saved in {}'.format(path))

        self.best_eval_path = path + '_best_auprc.pt'
        self.final_path = path + '_final.pt'

        self.criterion = nn.BCEWithLogitsLoss()
        if args.target == 'dx_depth1_unique':
            output_size = 18
        else:
            output_size = 1

        if args.transformer:
            print('Transformer')
            self.model = Transformer(args, output_size, device, target_file=args.source_file, n_layer=args.transformer_layers,
                                     attn_head=args.transformer_attn_heads, hidden_dim=args.transformer_hidden_dim).to(self.device)
        elif not args.transformer:
            if args.bert_induced:
                self.model = dict_post_RNN(args, output_size, device, target_file=args.source_file).to(self.device)
            elif not args.bert_induced:
                if args.item == 'lab':
                    vocab_size = 14371 if args.concat else 448
                elif args.item == 'med':
                    vocab_size = 4898 if args.concat else 2812
                elif args.item == 'inf':
                    vocab_size = 979
                elif args.item == 'all':
                    vocab_size = 15794 if args.concat else 3672
                self.model = RNNmodels(args=args, vocab_size=vocab_size, output_size=output_size, device=device).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        if args.lr_scheduler is not None:
            if args.lr_scheduler == 'labmda30':
                lambda1 = lambda epoch: 0.95 ** (epoch/30)
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=[lambda1])

            elif args.lr_scheduler == 'lambda20':
                lambda1 = lambda epoch: 0.90 ** (epoch/20)
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=[lambda1])

            elif args.lr_scheduler == 'plateau':
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', min_lr=1e-6)

        self.early_stopping = EarlyStopping(patience=100, verbose=True)






