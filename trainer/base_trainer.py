import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
import wandb
import os
import tqdm

from models.rnn_models import *
from utils.loss import *
from utils.trainer_utils import *

class Trainer(nn.Module):
    def __init__(self, args, train_dataloader, valid_dataloader, device):
        super().__init__()

        self.dataloader = train_dataloader
        self.eval_dataloader = valid_dataloader
        self.device = device

        wandb.init(project='learnable_cls_output', entity="pretrained_ehr", config=args, reinit=True)

        lr = args.lr
        self.n_epochs = args.n_epochs

        file_target_name = args.target
        if file_target_name == 'los>3day':
            file_target_name = 'los_3days'
        elif file_target_name == 'los>7day':
            file_target_name = 'los_7days'

        filename = 'trained_single_rnn_{}'.format(args.seed)
        path = os.path.join(args.path, 'singleRNN', args.source_file, file_target_name, filename)
        print('Model will be saved in {}'.format(path))

        self.best_eval_path = path + '_best_auprc.pt'
        self.final_path = path + '_final.pt'

        if args.source_file == 'mimic':
            vocab_size = 545
        elif args.source_file == 'eicu':
            vocab_size = 157
        else:
            raise NotImplementedError

        if args.target == 'dx_depth1_unique':
            output_size = 18
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            output_size = 1
            self.criterion = FocalLoss()

        self.model = RNNmodels(args=args, vocab_size=vocab_size, output_size=output_size, device=device).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.early_stopping = EarlyStopping(patience=30, verbose=True)
        num_params = count_parameters(self.model)
        print('Number of parameters: {}'.format(num_params))


    def train(self):
        best_loss = float('inf')
        best_auroc = 0.0
        best_auprc = 0.0

        # if os.path.exists(self.best_eval_path):
        #     ckpt = torch.load(self.best_eval_path)
        #     self.model.load_state_dict(ckpt['model_state_dict'])
        #     best_loss = ckpt['loss']
        #     best_auroc = ckpt['auroc']
        #     best_auprc = ckpt['auprc']

        for n_epoch in range(self.n_epochs + 1):

            preds_train = []
            truths_train = []
            avg_train_loss = 0.

            for iter, sample in tqdm.tqdm(enumerate(self.dataloader)):
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                item_id, target, lengths = sample
                item_id = item_id.to(self.device)
                target = target.to(self.device)

                y_pred = self.model(item_id, lengths)
                loss = self.criterion(y_pred, target.float().to(self.device))

                loss.backward()
                self.optimizer.step()

                avg_train_loss += loss.item() / len(self.dataloader)

                probs_train = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_train += list(probs_train.flatten())
                truths_train += list(target.detach().cpu().numpy().flatten())

            auroc_train = roc_auc_score(truths_train, preds_train)
            auprc_train = average_precision_score(truths_train, preds_train)

            avg_eval_loss, auroc_eval, auprc_eval = self.evaluation()

            if best_auprc < auprc_eval:
                best_loss = avg_eval_loss
                best_auroc = auroc_eval
                best_auprc = auprc_eval

                torch.save({'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': best_loss,
                            'auroc': best_auroc,
                            'auprc': best_auprc,
                            'epochs': n_epoch}, self.best_eval_path)

                print('Model parameter saved at epoch {}'.format(n_epoch))

            wandb.log({'train_loss': avg_train_loss,
                       'train_auroc': auroc_train,
                       'train_auprc': auprc_train,
                       'eval_loss': avg_eval_loss,
                       'eval_auroc': auroc_eval,
                       'eval_auprc': auprc_eval})

            print('[Train]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_train_loss, auroc_train, auprc_train))
            print('[Valid]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_eval_loss, auroc_eval, auprc_eval))

            self.early_stopping(auprc_eval)
            if self.early_stopping.early_stop:
                print('Early stopping')
                torch.save({'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': avg_eval_loss,
                            'auroc': auroc_eval,
                            'auprc': auprc_eval,
                            'epochs': n_epoch}, self.final_path)
                break


    def evaluation(self):
        self.model.eval()

        preds_eval = []
        truths_eval = []
        avg_eval_loss = 0.

        with torch.no_grad():
            for iter, sample in enumerate(self.eval_dataloader):
                item_id, target, lengths  = sample
                item_id.to(self.device); target.to(self.device)

                y_pred = self.model(item_id, lengths)
                loss = self.criterion(y_pred, target.float().to(self.device))
                avg_eval_loss += loss.item() / len(self.eval_dataloader)

                probs_eval = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_eval += list(probs_eval.flatten())
                truths_eval += list(target.detach().cpu().numpy().flatten())

            auroc_eval = roc_auc_score(truths_eval, preds_eval)
            auprc_eval = average_precision_score(truths_eval, preds_eval)

        return avg_eval_loss, auroc_eval, auprc_eval









