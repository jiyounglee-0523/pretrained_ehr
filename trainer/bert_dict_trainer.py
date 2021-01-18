import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
import wandb
import os
import tqdm

from models.rnn_bert_dict import dict_post_RNN
from utils.loss import *
from utils.trainer_utils import *

class bert_dict_Trainer():
    def __init__(self, args, train_dataloader, valid_dataloader, device):
        self.dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.device = device
        self.debug = args.debug

        if not self.debug:
            wandb.init(project='comparison-between-berts', entity="pretrained_ehr", config=args, reinit=True)

        lr = args.lr
        self.n_epochs = args.n_epochs

        file_target_name = args.target
        if file_target_name == 'los>3day':
            file_target_name = 'los_3days'
        elif file_target_name == 'los>7day':
            file_target_name = 'los_7days'

        if args.cls_freeze:
            if args.concat:
                if args.only_BCE:
                    filename = 'cls_fixed_{}_{}_concat_onlyBCE'.format(args.bert_model, args.seed)
                elif not args.only_BCE:
                    filename = 'cls_fixed_{}_{}_concat'.format(args.bert_model, args.seed)
            elif not args.concat:
                if args.only_BCE:
                    filename = 'cls_fixed_{}_{}_onlyBCE'.format(args.bert_model, args.seed)
                elif not args.only_BCE:
                    filename = 'cls_fixed_{}_{}'.format(args.bert_model, args.seed)

        elif not args.cls_freeze:
            if args.concat:
                if args.only_BCE:
                    filename = 'cls_learnable_{}_{}_concat_onlyBCE'.format(args.bert_model, args.seed)
                elif not args.only_BCE:
                    filename = 'cls_learnable_{}_{}_concat'.format(args.bert_model, args.seed)
            elif not args.concat:
                if args.only_BCE:
                    filename = 'cls_learnable_{}_{}_onlyBCE'.format(args.bert_model, args.seed)
                elif not args.only_BCE:
                    filename = 'cls_learnable_{}_{}'.format(args.bert_model, args.seed)

        path = os.path.join(args.path, args.item, 'cls_learnable', args.source_file, file_target_name, filename)
        print('Model will be saved in {}'.format(path))

        self.best_eval_path = path + '_best_auprc.pt'
        self.final_path = path + '_final.pt'

        if args.only_BCE:
            self.criterion = nn.BCEWithLogitsLoss()
            if args.target == 'dx_depth1_unique':
                output_size = 18
            else:
                output_size = 1
        elif not args.only_BCE:
            if args.target == 'dx_depth1_unique':
                output_size = 18
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                output_size = 1
                self.criterion = FocalLoss()

        self.model = dict_post_RNN(args, output_size, device, target_file=args.source_file).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.early_stopping = EarlyStopping(patience=30, verbose=True)
        num_params = count_parameters(self.model)
        print('Number of parameters: {}'.format(num_params))

    def train(self):
        best_loss = float('inf')
        best_auroc = 0.0
        best_auprc = 0.0

        for n_epoch in range(self.n_epochs + 1):
            preds_train = []
            truths_train = []
            avg_train_loss = 0.0

            for iter, sample in tqdm.tqdm(enumerate(self.dataloader)):
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                item_embed, item_target, seq_len = sample
                item_embed = item_embed.to(self.device)
                item_target = item_target.to(self.device)

                y_pred = self.model(item_embed, seq_len)
                loss = self.criterion(y_pred, item_target.float().to(self.device))

                loss.backward()
                self.optimizer.step()

                avg_train_loss += loss.item() / len(self.dataloader)

                probs_train = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_train += list(probs_train.flatten())
                truths_train += list(item_target.detach().cpu().numpy().flatten())

            auroc_train = roc_auc_score(truths_train, preds_train)
            auprc_train = average_precision_score(truths_train, preds_train)

            avg_eval_loss, auroc_eval, auprc_eval = self.evaluation()

            if best_auprc < auprc_eval:
                best_loss = avg_eval_loss
                best_auroc = auroc_eval
                best_auprc = auprc_eval
                if not self.debug:
                    torch.save({'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': best_loss,
                                'auroc': best_auroc,
                                'auprc': best_auprc,
                                'epochs': n_epoch}, self.best_eval_path)
                    print('Model parameter saved at epoch {}'.format(n_epoch))

            if not self.debug:
                wandb.log({'train_loss': avg_train_loss,
                           'train_auroc': auroc_train,
                           'train_auprc': auprc_train,
                           'eval_loss': avg_eval_loss,
                           'eval_auroc': auroc_eval,
                           'eval_auprc': auprc_eval})

            print('[Train]  loss: {:.3f},  auroc: {:.3f},   auprc: {:.3f}'.format(avg_train_loss, auroc_train, auprc_train))
            print('[Valid]  loss: {:.3f},  auroc: {:.3f},   auprc: {:.3f}'.format(avg_eval_loss, auroc_eval, auprc_eval))

            self.early_stopping(auprc_eval)

            if self.early_stopping.early_stop:
                print('Early stopping')
                if not self.debug:
                    torch.save({'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': avg_eval_loss,
                                'auroc': best_auroc,
                                'auprc': best_auprc,
                                'epochs': n_epoch}, self.final_path)
                break

    def evaluation(self):
        self.model.eval()

        preds_eval = []
        truths_eval = []
        avg_eval_loss = 0.0

        with torch.no_grad():
            for iter, sample in enumerate(self.valid_dataloader):
                item_embed, item_target, seq_len = sample
                item_embed = item_embed.to(self.device)
                item_target = item_target.to(self.device)

                y_pred = self.model(item_embed, seq_len)
                loss = self.criterion(y_pred, item_target.float().to(self.device))
                avg_eval_loss += loss.item() / len(self.valid_dataloader)

                probs_eval = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_eval += list(probs_eval.flatten())
                truths_eval += list(item_target.detach().cpu().numpy().flatten())

            auroc_eval = roc_auc_score(truths_eval, preds_eval)
            auprc_eval = average_precision_score(truths_eval, preds_eval)

        return avg_eval_loss, auroc_eval, auprc_eval
