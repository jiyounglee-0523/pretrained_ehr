import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
import wandb
import os
import tqdm

from models.prebert import *
from models.transformer import Transformer
from utils.loss import *
from utils.trainer_utils import *
from dataset.prebert_dataloader import bertinduced_get_dataloader

class Bert_Trainer():
    def __init__(self, args, train_dataloader, valid_dataloader, device):
        self.dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        if args.source_file != 'both':
            self.test_dataloader = bertinduced_get_dataloader(args, data_type='test')
        elif args.source_file == 'both':
            self.mimic_test_dataloader = bertinduced_get_dataloader(args, data_type='test', data_name='mimic')
            self.eicu_test_dataloader = bertinduced_get_dataloader(args, data_type='test', data_name='eicu')
        self.device = device
        self.debug = args.debug
        self.BCE = args.only_BCE
        self.target = args.target
        self.source_file = args.source_file

        if not self.debug:
            wandb.init(project="bert_finetune", entity="pretrained_ehr", config=args, reinit=True)

        lr = args.lr
        self.n_epochs = args.n_epochs

        file_target_name = args.target
        if file_target_name == 'los>3day':
            file_target_name = 'los_3days'
        elif file_target_name == 'los>7day':
            file_target_name = 'los_7days'

        if not args.transformer:
            if args.only_BCE:
                filename = 'bert_finetune_{}_rnn_{}_onlyBCE'.format(args.bert_model, args.seed)
            elif not args.only_BCE:
                filename = 'bert_finetune_bertfreeze_{}_rnn_{}'.format(args.bert_model, args.seed)
        elif args.transformer:
            if args.only_BCE:
                filename = 'bert_finetune_{}_transformer_{}_onlyBCE'.format(args.bert_model, args.seed)
            elif not args.only_BCE:
                filename = 'bert_finetune_{}_transformer_{}'.format(args.bert_model, args.seed)

        path = os.path.join(args.path, args.item, 'bert_finetune', args.source_file, file_target_name, filename)
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
        if args.transformer:
            self.model = nn.DataParallel(post_Transformer(args, output_size, device, n_layers=args.transformer_layers,
                                                     attn_head=args.transformer_attn_heads, hidden_dim=args.transformer_hidden_dim)).to(self.device)
        elif not args.transformer:
            self.model = nn.DataParallel(post_RNN(args, output_size)).to(self.device)
            self.model.to(self.device)

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
            avg_train_loss = 0.0

            for iter, sample in tqdm.tqdm(enumerate(self.dataloader)):
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                item_name, item_target, offset_order, masking = sample
                #item_name = item_name.to(self.device)
                item_target = item_target.to(self.device)
                # print('DataLoader Done!')
                # print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
                y_pred = self.model(item_name, offset_order, masking)
                if self.BCE and self.target != 'dx_depth1_unique':
                    loss = self.criterion(y_pred, item_target.unsqueeze(1).float().to(self.device))
                else:
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

            if not self.debug:
                wandb.log({'train_loss': avg_train_loss,
                           'train_auroc': auroc_train,
                           'train_auprc': auprc_train,
                           'eval_loss': avg_eval_loss,
                           'eval_auroc': auroc_eval,
                           'eval_auprc': auprc_eval})

            if best_auprc < auprc_eval:
                best_loss = avg_eval_loss
                best_auroc = auroc_eval
                best_auprc = auprc_eval
                if not self.debug:
                    torch.save({'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': avg_eval_loss,
                                'auroc': best_auroc,
                                'auprc': best_auprc,
                                'epochs': n_epoch}, self.best_eval_path)
                    print('Model parameter saved at epoch {}'.format(n_epoch))

            print('[Train_{}]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(n_epoch, avg_train_loss, auroc_train, auprc_train))
            print('[Valid_{}]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(n_epoch, avg_eval_loss, auroc_eval, auprc_eval))

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
                if self.source_file == 'both':
                    self.test_both()
                else:
                    self.test()
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

                if self.BCE and self.target != 'dx_depth1_unique':
                    loss = self.criterion(y_pred, item_target.unsqueeze(1).float().to(self.device))
                else:
                    loss = self.criterion(y_pred, item_target.float().to(self.device))
                avg_eval_loss += loss.item() / len(self.valid_dataloader)

                probs_eval = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_eval += list(probs_eval.flatten())
                truths_eval += list(item_target.detach().cpu().numpy().flatten())

            auroc_eval = roc_auc_score(truths_eval, preds_eval)
            auprc_eval = average_precision_score(truths_eval, preds_eval)

        return avg_eval_loss, auroc_eval, auprc_eval

    def test(self):
        ckpt = torch.load(self.best_eval_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        preds_test = []
        truths_test = []
        avg_test_loss = 0.

        with torch.no_grad():
            for iter, sample in enumerate(self.test_dataloader):
                item_name, item_target, seq_len = sample
                item_target = item_target.to(self.device)

                y_pred = self.model(item_name, seq_len)
                if self.BCE and self.target != 'dx_depth1_unique':
                    loss = self.criterion(y_pred, item_target.unsqueeze(1).float().to(self.device))
                else:
                    loss = self.criterion(y_pred, item_target.float().to(self.device))
                avg_test_loss += loss.item() / len(self.test_dataloader)

                probs_test = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_test += list(probs_test.flatten())
                truths_test += list(item_target.detach().cpu().numpy().flatten())

            auroc_test = roc_auc_score(truths_test, preds_test)
            auprc_test = average_precision_score(truths_test, preds_test)

            if not self.debug:
                wandb.log({'test_loss': avg_test_loss,
                           'test_auroc': auroc_test,
                           'test_auprc': auprc_test})
            print('[Test]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_test_loss, auroc_test, auprc_test))

    def test_both(self):
        ckpt = torch.load(self.best_eval_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        preds_test = []
        truths_test = []
        avg_test_loss = 0.

        with torch.no_grad():
            for iter, sample in enumerate(self.mimic_test_dataloader):
                item_id, item_target, seq_len = sample
                item_target.to(self.device)

                y_pred = self.model(item_id, seq_len)
                if self.BCE and self.target != 'dx_depth1_unique':
                    loss = self.criterion(y_pred, item_target.unsqueeze(1).float().to(self.device))
                else:
                    loss = self.criterion(y_pred, item_target.float().to(self.device))
                avg_test_loss += loss.item() / len(self.mimic_test_dataloader)

                probs_test = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_test += list(probs_test.flatten())
                truths_test += list(item_target.detach().cpu().numpy().flatten())

            auroc_test = roc_auc_score(truths_test, preds_test)
            auprc_test = average_precision_score(truths_test, preds_test)

            if not self.debug:
                wandb.log({'mimic_test_loss': avg_test_loss,
                           'mimic_test_auroc': auroc_test,
                           'mimic_test_auprc': auprc_test})

            print('[Test/mimic]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_test_loss, auroc_test, auprc_test))

        preds_test = []
        truths_test = []
        avg_test_loss = 0.

        with torch.no_grad():
            for iter, sample in enumerate(self.eicu_test_dataloader):
                item_id, item_target, seq_len = sample
                item_target.to(self.device)

                y_pred = self.model(item_id, seq_len)
                if self.BCE and self.target != 'dx_depth1_unique':
                    loss = self.criterion(y_pred, item_target.unsqueeze(1).float().to(self.device))
                else:
                    loss = self.criterion(y_pred, item_target.float().to(self.device))
                avg_test_loss += loss.item() / len(self.eicu_test_dataloader)

                probs_test = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_test += list(probs_test.flatten())
                truths_test += list(item_target.detach().cpu().numpy().flatten())

            auroc_test = roc_auc_score(truths_test, preds_test)
            auprc_test = average_precision_score(truths_test, preds_test)

            if not self.debug:
                wandb.log({'eicu_test_loss': avg_test_loss,
                           'eicu_test_auroc': auroc_test,
                           'eicu_test_auprc': auprc_test})

            print('[Test/eicu]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_test_loss, auroc_test, auprc_test))



