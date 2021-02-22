import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
import os
import tqdm

from models.DescEmb import DescEmb
from utils.loss import *
from utils.trainer_utils import *
from dataset.DescEmb_dataloader import DescEmb_get_dataloader

class DescEmb_Trainer():
    def __init__(self, args, train_dataloader, device):
        self.dataloader = train_dataloader
        self.test_dataloader = DescEmb_get_dataloader(args, data_type='test')
        self.valid_dataloader = DescEmb_get_dataloader(args, data_type='eval')

        self.device = device
        self.target = args.target
        self.source_file = args.source_file

        lr = args.lr
        self.n_epochs = args.n_epochs

        file_target_name = args.target
        if file_target_name == 'los>3day':
            file_target_name = 'los_3days'
        elif file_target_name == 'los>7day':
            file_target_name = 'los_7days'

        if args.cls_freeze:    # DescEmb-FR
            filename = 'cls_fixed_{}_{}'.format(args.bert_model, args.seed)

        elif not args.cls_freeze:   # DescEmb-FT
            filename = 'cls_learnable_{}_{}'.format(args.bert_model, args.seed)


        path = os.path.join(args.path, args.item, 'cls_learnable', args.source_file, file_target_name, filename)
        print('Model will be saved in {}'.format(path))

        self.best_eval_path = path + '_best_auprc.pt'


        self.criterion = nn.BCEWithLogitsLoss()
        if args.target == 'dx_depth1_unique':
            output_size = 18
        else:
            output_size = 1

        self.model = DescEmb(args, output_size, device, target_file=args.source_file).to(self.device)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)


        self.early_stopping = EarlyStopping(patience=50, verbose=True)
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

                if self.target != 'dx_depth1_unique':
                    loss = self.criterion(y_pred, item_target.unsqueeze(1).float().to(self.device))
                else:
                    loss = self.criterion(y_pred, item_target.float().to(self.device))

                loss.backward()
                self.optimizer.step()

                avg_train_loss += loss.item() / len(self.dataloader)

                probs_train = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_train += list(probs_train.flatten())
                truths_train += list(item_target.detach().cpu().numpy().flatten())

            auroc_train = roc_auc_score(truths_train, preds_train)
            auprc_train = average_precision_score(truths_train, preds_train, average='micro')

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

            print('[Train]  loss: {:.3f},  auroc: {:.3f},   auprc: {:.3f}'.format(avg_train_loss, auroc_train,
                                                                                  auprc_train))
            print('[Valid]  loss: {:.3f},  auroc: {:.3f},   auprc: {:.3f}'.format(avg_eval_loss, auroc_eval,
                                                                                  auprc_eval))

            self.early_stopping(auprc_eval)
            if self.early_stopping.early_stop:
                print('Early stopping')
                break


        if self.source_file != 'both':
            self.test()


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

                if self.target != 'dx_depth1_unique':
                    loss = self.criterion(y_pred, item_target.unsqueeze(1).float().to(self.device))
                else:
                    loss = self.criterion(y_pred, item_target.float().to(self.device))
                avg_eval_loss += loss.item() / len(self.valid_dataloader)

                probs_eval = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_eval += list(probs_eval.flatten())
                truths_eval += list(item_target.detach().cpu().numpy().flatten())

            auroc_eval = roc_auc_score(truths_eval, preds_eval)
            auprc_eval = average_precision_score(truths_eval, preds_eval, average='micro')

        return avg_eval_loss, auroc_eval, auprc_eval


    def test(self):
        ckpt = torch.load(self.best_eval_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        preds_test = []
        truths_test = []
        avg_test_loss =0.

        with torch.no_grad():
            for iter, sample in enumerate(self.test_dataloader):
                item_embed, item_target, seq_len = sample
                item_embed = item_embed.to(self.device)
                item_target = item_target.to(self.device)
                y_pred = self.model(item_embed, seq_len)
                if self.target != 'dx_depth1_unique':
                    loss = self.criterion(y_pred, item_target.unsqueeze(1).float().to(self.device))
                else:
                    loss = self.criterion(y_pred, item_target.float().to(self.device))
                avg_test_loss += loss.item() / len(self.test_dataloader)

                probs_test = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_test += list(probs_test.flatten())
                truths_test += list(item_target.detach().cpu().numpy().flatten())

            auroc_test = roc_auc_score(truths_test, preds_test)
            auprc_test = average_precision_score(truths_test, preds_test, average='micro')


            print('[Test]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_test_loss, auroc_test, auprc_test))
