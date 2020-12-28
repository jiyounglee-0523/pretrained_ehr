import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
import wandb

from models.rnn_models import *
from utils.loss import *

class Trainer(nn.Module):
    def __init__(self, args, train_dataloader, valid_dataloader, device):
        super().__init__()

        self.dataloader = train_dataloader
        self.eval_dataloader = valid_dataloader
        self.criterion = FocalLoss()
        self.device = device

        lr = args.lr
        self.n_epochs = args.n_epochs
        self.path = './test.pt'      ########## change this!

        if args.source_file == 'mimic':
            vocab_size = 600             ########### change this!   vocab size 잘못됨
        elif args.source_file == 'eicu':
            vocab_size = 150
        else:
            raise NotImplementedError

        if args.target == 'diagnosis':
            output_size = 17
        else:
            output_size = 1

        self.model = RNNmodels(args=args, vocab_size=vocab_size, output_size=output_size, device=device).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # wandb.init(project='pretrained_ehr')
        # wandb.config.update(args)

    def train(self):
        best_loss = float('inf')
        best_auroc = 0.0
        best_auprc = 0.0
        for n_epoch in range(self.n_epochs + 1):

            preds_train = []
            truths_train = []
            avg_train_loss = 0.

            for iter, sample in enumerate(self.dataloader):
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                item_id, item_offset, item_offset_order, lengths, target = sample
                item_id.to(self.device) ; item_offset.to(self.device) ; item_offset_order.to(self.device)
                lengths.to(self.device) ; target.to(self.device)

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

            # wandb.log({'train_loss': avg_train_loss,
            #            'train_auroc': auroc_train,
            #            'train_auprc': auprc_train,
            #            'eval_loss': avg_eval_loss,
            #            'eval_auroc': auroc_eval,
            #            'eval_auprc': auprc_eval})

            if n_epoch % 20 == 0:
                print('[Train]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_train_loss, auroc_train, auprc_train))
                print('[Valid]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_eval_loss, auroc_eval, auprc_eval))



    def evaluation(self):
        self.model.eval()

        preds_eval = []
        truths_eval = []
        avg_eval_loss = 0.

        with torch.no_grad():
            for iter, sample in enumerate(self.eval_dataloader):
                item_id, item_offset, item_offset_order, lengths, target = sample
                item_id.to(self.device);
                item_offset.to(self.device);
                item_offset_order.to(self.device)
                lengths.to(self.device);
                target.to(self.device)

                y_pred = self.model(item_id, lengths)
                loss = self.criterion(y_pred, target.float().to(self.device))
                avg_eval_loss += loss.item() / len(self.eval_dataloader)

                probs_eval = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_eval += list(probs_eval.flatten())
                truths_eval += list(target.detach().cpu().numpy().flatten())

            auroc_eval = roc_auc_score(truths_eval, preds_eval)
            auprc_eval = average_precision_score(truths_eval, preds_eval)

            return avg_eval_loss, auroc_eval, auprc_eval









