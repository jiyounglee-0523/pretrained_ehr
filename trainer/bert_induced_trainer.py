import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
import wandb
import os
import tqdm

from models.prebert import *
from utils.loss import *
from utils.trainer_utils import *

class Bert_Trainer():
    def __init__(self, args, train_dataloader, valid_dataloader, device):

        self.dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.device = device

        wandb.init(project="fine_tuning_test", entity="pretrained_ehr", config=args, reinit=True)
        args = wandb.config

        lr = args.lr
        self.n_epochs = args.n_epochs

        file_target_name = args.target
        if file_target_name == 'los>3day':
            file_target_name = 'los_3day'
        elif file_target_name == 'los>7day':
            file_target_name = 'los_7day'

        if args.only_BCE:
            filename = 'bert_finetune_{}_onlyBCE'.format(args.seed)
        elif not args.only_BCE:
            filename = 'bert_finetune_{}'.format(args.seed)

        path = os.path.join(args.path, 'bert_finetune', args.source_file, file_target_name, filename)
        print('Model will be saved in {}'.format(path))

        self.best_eval_path = path + '_best_eval.pt'
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

        self.model = nn.DataParallel(post_RNN(args, output_size, self.criterion))
        self.model.to(self.device)
        #self.model = nn.DataParallel(post_RNN(args, output_size, self.criterion).to(self.device))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.early_stopping = EarlyStopping(patience = 30, verbose=True)

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
            avg_train_loss = 0.0

            for iter, sample in tqdm.tqdm(enumerate(self.dataloader)):
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                item_name, item_target, seq_len = sample
                item_target = item_target.to(self.device)
                # print('DataLoader Done!')
                # print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
                y_pred = self.model(item_name, seq_len)
                loss = self.criterion(y_pred, item_target.float())
                #y_pred, loss = self.model(item_name, seq_len, item_target)
                #loss = torch.mean(loss)
                # print('Forward Done!')

                loss.backward()
                # print('Backward Done!')
                self.optimizer.step()

                # print('Model parameters')
                #
                # for name, param in self.model.prebert.named_parameters():
                #     #if name == 'model.embeddings.word_embeddings.weight':
                #     print(name, param.data)
                # print('====================================')

                avg_train_loss += loss.item() / len(self.dataloader)

                probs_train = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_train += list(probs_train.flatten())
                truths_train += list(item_target.detach().cpu().numpy().flatten())
                #wandb.log({'train_loss': loss})

            auroc_train = roc_auc_score(truths_train, preds_train)
            auprc_train = average_precision_score(truths_train, preds_train)

            avg_eval_loss, auroc_eval, auprc_eval = self.evaluation()

            wandb.log({'train_loss': avg_train_loss,
                       'train_auroc': auroc_train,
                       'train_auprc': auprc_train,
                       'eval_loss': avg_eval_loss,
                       'eval_auroc': auroc_eval,
                       'eval_auprc': auprc_eval})

            if best_auprc > auprc_eval:
                best_loss = avg_eval_loss
                best_auroc = auroc_eval
                best_auprc = auprc_eval
                torch.save({'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': avg_eval_loss,
                            'auroc': best_auroc,
                            'auprc': best_auprc,
                            'epochs': n_epoch}, self.best_eval_path)
                print('Model parameter saved at epoch {}'.format(n_epoch))

            print('[Train_{}]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(n_epoch, avg_train_loss,
                                                                                            auroc_train, auprc_train))
            print('[Valid_{}]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(n_epoch, avg_eval_loss,
                                                                                            auroc_eval, auprc_eval))

            self.early_stopping(auprc_eval)
            if self.early_stopping.early_stop:
                print('Early stopping')
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
                item_name, item_target, seq_len = sample
                item_target = item_target.to(self.device)

                y_pred = self.model(item_name, seq_len)
                loss = self.criterion(y_pred, item_target.float().to(self.device))
                avg_eval_loss += loss.item() / len(self.valid_dataloader)

                probs_eval = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_eval += list(probs_eval.flatten())
                truths_eval += list(item_target.detach().cpu().numpy().flatten())

            auroc_eval = roc_auc_score(truths_eval, preds_eval)
            auprc_eval = average_precision_score(truths_eval, preds_eval)

        return avg_eval_loss, auroc_eval, auprc_eval




