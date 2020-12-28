import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
import wandb
import os

from models.prebert import *
from utils.loss import *

class Bert_Trainer():
    def __init__(self, args, train_dataloader, valid_dataloader, device):
        super().__init__()

        self.dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.criterion = FocalLoss()
        self.device = device

        lr = args.lr
        self.n_epochs = args.n_epochs
        path = args.path
        filename = args.filename
        self.path = path + filename + '.pt'

        if args.target == 'diagnosis':
            output_size = 17
        else:
            output_size = 1

        self.model = post_RNN(args, output_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        wandb.init(project='pretrained_ehr')
        wandb.config.update(args)

    def train(self):
        best_loss = float('inf')
        best_auroc = 0.0
        best_auprc = 0.0

        if os.path.exists(self.path):
            ckpt = torch.load(self.path)
            self.model.load_state_dict(ckpt['model_state_dict'])
            best_loss = ckpt['loss']
            best_auroc = ckpt['auroc']
            best_auprc = ckpt['auprc']

        for n_epoch in range(self.n_epochs + 1):
            preds_train = []
            truths_train = []
            avg_train_loss = 0.0

            for iter, sample in enumerate(self.dataloader):
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                item_name, item_target, seq_len = sample
                item_target = item_target.to(self.device) ; seq_len = seq_len.to(self.device)

                y_pred = self.model(item_name, seq_len)
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

            wandb.log({'train_loss': avg_train_loss,
                       'train_auroc': auroc_train,
                       'train_auprc': auprc_train,
                       'eval_loss': avg_eval_loss,
                       'eval_auroc': auroc_eval,
                       'eval_auprc': auprc_eval})

            if best_loss > avg_eval_loss:
                best_loss = avg_eval_loss
                best_auroc = auroc_eval
                best_auprc = auprc_eval
                torch.save({'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': avg_eval_loss,
                            'auroc': best_auroc,
                            'auprc': best_auprc}, self.path)
                print('Model parameter saved at epoch {}'.format(n_epoch))


            print('[Train_{}]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(n_epoch, avg_train_loss, auroc_train, auprc_train))
            print('[Valid_{}]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(n_epoch, avg_eval_loss, auroc_eval, auprc_eval))


    def evaluation(self):
        self.model.eval()

        preds_eval = []
        truths_eval = []
        avg_eval_loss = 0.0

        with torch.no_grad():
            for iter, sample in enumerate(self.valid_dataloader):
                item_name, item_target, seq_len = sample
                item_name = item_name.to(self.device);
                item_target = item_target.to(self.device);
                seq_len = seq_len.to(self.device)

                y_pred = self.model(item_name, seq_len)
                loss = self.criterion(y_pred, item_target.float().to(self.device))
                avg_eval_loss += loss.item() / len(self.valid_dataloader)

                probs_eval = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_eval += list(probs_eval.flatten())
                truths_eval += list(item_target.detach().cpu().numpy().flatten())

            auroc_eval = roc_auc_score(truths_eval, preds_eval)
            auprc_eval = average_precision_score(truths_eval, preds_eval)

        return avg_eval_loss, auroc_eval, auprc_eval




