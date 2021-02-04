import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
import wandb
import os
import tqdm

from models.rnn_bert_dict import dict_post_RNN
from models.transformer import Transformer
from utils.loss import *
from utils.trainer_utils import *
from dataset.prebert_dict_dataloader import bertinduced_dict_get_dataloader

class bert_dict_Trainer():
    def __init__(self, args, train_dataloader, device):
        self.dataloader = train_dataloader
        if args.source_file != 'both':
            self.test_dataloader = bertinduced_dict_get_dataloader(args, data_type='test')
            self.valid_dataloader = bertinduced_dict_get_dataloader(args, data_type='eval')
        elif args.source_file == 'both':
            self.mimic_test_dataloader = bertinduced_dict_get_dataloader(args, data_type='test', data_name='mimic')
            self.mimic_valid_dataloader = bertinduced_dict_get_dataloader(args, data_type='eval', data_name='mimic')
            self.eicu_test_dataloader = bertinduced_dict_get_dataloader(args, data_type='test', data_name='eicu')
            self.eicu_valid_dataloader = bertinduced_dict_get_dataloader(args, data_type='eval', data_name='eicu')

        self.device = device
        self.debug = args.debug
        self.BCE = args.only_BCE
        self.target = args.target
        self.source_file = args.source_file
        self.lr_scheduler = args.lr_scheduler

        if not self.debug:
            wandb.init(project=args.wandb_project_name, entity="pretrained_ehr", config=args, reinit=True)

        lr = args.lr
        self.lr = lr
        self.n_epochs = args.n_epochs

        file_target_name = args.target
        if file_target_name == 'los>3day':
            file_target_name = 'los_3days'
        elif file_target_name == 'los>7day':
            file_target_name = 'los_7days'

        if not args.transformer:
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
                        filename = 'cls_learnable_{}_{}_{}_{}_onlyBCE'.format(args.bert_model, args.seed, args.lr_scheduler, args.lr)
                    elif not args.only_BCE:
                        filename = 'cls_learnable_{}_{}'.format(args.bert_model, args.seed)
        elif args.transformer:
            if args.cls_freeze:
                if args.concat:
                    if args.only_BCE:
                        filename = 'cls_fixed_transformer_layers{}_attnheads{}_hidden{}_{}_{}_{}_concat_onlyBCE'.format(args.transformer_layers, args.transformer_attn_heads,
                                                                                                                     args.transformer_hidden_dim, args.bert_model, args.seed, args.max_length)
                    elif not args.only_BCE:
                        filename = 'cls_fixed_transformer_layers{}_attnheads{}_hidden{}_{}_{}_{}_concat'.format(args.transformer_layers, args.transformer_attn_heads,
                                                                                                                     args.transformer_hidden_dim, args.bert_model, args.seed, args.max_length)
                elif not args.concat:
                    if args.only_BCE:
                        filename = 'cls_fixed_transformer_layers{}_attnheads{}_hidden{}_{}_{}_{}_onlyBCE'.format(args.transformer_layers, args.transformer_attn_heads,
                                                                                                                     args.transformer_hidden_dim, args.bert_model, args.seed, args.max_length)
                    elif not args.only_BCE:
                        filename = 'cls_fixed_transformer_layers{}_attnheads{}_hidden{}_{}_{}_{}'.format(args.transformer_layers, args.transformer_attn_heads,
                                                                                                                     args.transformer_hidden_dim, args.bert_model, args.seed, args.max_length)
            elif not args.cls_freeze:
                if args.concat:
                    if args.only_BCE:
                        filename = 'cls_learnable_transformer_layers{}_attnheads{}_hidden{}_{}_{}_{}_concat_onlyBCE'.format(args.transformer_layers, args.transformer_attn_heads,
                                                                                                                     args.transformer_hidden_dim, args.bert_model, args.seed, args.max_length)
                    elif not args.only_BCE:
                        filename = 'cls_learnable_transformer_layers{}_attnheads{}_hidden{}_{}_{}_{}_concat'.format(args.transformer_layers, args.transformer_attn_heads,
                                                                                                                     args.transformer_hidden_dim, args.bert_model, args.seed, args.max_length)
                elif not args.concat:
                    if args.only_BCE:
                        filename = 'cls_learnable_transformer_layers{}_attnheads{}_hidden{}_{}_{}_{}_{}_{}_{}_onlyBCE'.format(args.transformer_layers, args.transformer_attn_heads,
                                                                                                                     args.transformer_hidden_dim, args.bert_model, args.seed, args.max_length, args.lr_scheduler, args.lr, args.batch_size)
                    elif not args.only_BCE:
                        filename = 'cls_learnable_transformer_layers{}_attnheads{}_hidden{}_{}_{}_{}'.format(args.transformer_layers, args.transformer_attn_heads,
                                                                                                                     args.transformer_hidden_dim, args.bert_model, args.seed, args.max_length)

        path = os.path.join(args.path, args.item, 'cls_learnable', args.source_file, file_target_name, filename)
        print('Model will be saved in {}'.format(path))

        self.best_eval_path = path + '_best_auprc.pt'
        self.best_mimic_eval_path = path + '_mimic_best_auprc.pt'
        self.best_eicu_eval_path = path + '_eicu_best_auprc.pt'
        self.final_path = path + '_final.pt'
        # self.mimic_final_path = path + '_mimic_final.pt'
        # self.eicu_final_path = path + '_eicu_final.pt'

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
            print('Transformer')
            self.model = Transformer(args, output_size, device, target_file=args.source_file, n_layer=args.transformer_layers,
                                     attn_head=args.transformer_attn_heads, hidden_dim=args.transformer_hidden_dim).to(self.device)
        elif not args.transformer:
            print('RNN')
            self.model = dict_post_RNN(args, output_size, device, target_file=args.source_file).to(self.device)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        if args.lr_scheduler is not None:
            if args.lr_scheduler == 'lambda30':
                lambda1 = lambda epoch: 0.95 ** (epoch/30)
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=[lambda1])

            elif args.lr_scheduler == 'lambda20':
                lambda1 = lambda epoch: 0.90 ** (epoch / 20)
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=[lambda1])

            elif args.lr_scheduler == 'plateau':
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', min_lr=1e-6)

        if args.source_file != 'both':
            self.early_stopping = EarlyStopping(patience=50, verbose=True)
        elif args.source_file == 'both':
            self.mimic_early_stopping = EarlyStopping(patience=50, verbose=True)
            self.eicu_early_stopping = EarlyStopping(patience=50, verbose=True)
        num_params = count_parameters(self.model)
        print('Number of parameters: {}'.format(num_params))

    def train(self):
        best_loss = float('inf')
        best_mimic_loss = float('inf')
        best_eicu_loss = float('inf')
        best_auroc = 0.0
        best_mimic_auroc = 0.0
        best_eicu_auroc = 0.0
        best_auprc = 0.0
        best_mimic_auprc = 0.0
        best_eicu_auprc = 0.0
        lr = self.lr

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

                if self.BCE and self.target != 'dx_depth1_unique':
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

            if self.source_file != 'both':
                avg_eval_loss, auroc_eval, auprc_eval = self.evaluation()

                if self.lr_scheduler is not None:
                    if self.lr_scheduler != 'plateau':
                        self.scheduler.step()
                    elif self.lr_scheduler == 'plateau':
                        self.scheduler.step(auprc_eval)
                    lr = self.optimizer.param_groups[0]['lr']

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

                if not self.debug:
                    wandb.log({'train_loss': avg_train_loss,
                               'train_auroc': auroc_train,
                               'train_auprc': auprc_train,
                               'eval_loss': avg_eval_loss,
                               'eval_auroc': auroc_eval,
                               'eval_auprc': auprc_eval,
                               'lr': lr})

                print('[Train]  loss: {:.3f},  auroc: {:.3f},   auprc: {:.3f}'.format(avg_train_loss, auroc_train,
                                                                                      auprc_train))
                print('[Valid]  loss: {:.3f},  auroc: {:.3f},   auprc: {:.3f}'.format(avg_eval_loss, auroc_eval,
                                                                                      auprc_eval))

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



            elif self.source_file == 'both':
                mimic_avg_eval_loss, mimic_auroc_eval, mimic_auprc_eval, eicu_avg_eval_loss, eicu_auroc_eval, eicu_auprc_eval = self.evaluation_both()

                if best_mimic_auprc < mimic_auprc_eval:
                    best_mimic_loss = mimic_avg_eval_loss
                    best_mimic_auroc = mimic_auroc_eval
                    best_mimic_auprc = mimic_auprc_eval
                    torch.save({'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': best_mimic_loss,
                                'auroc': best_mimic_auroc,
                                'auprc': best_mimic_auprc,
                                'epochs': n_epoch}, self.best_mimic_eval_path)
                    print('[mimic] Model parameter saved at epoch {}'.format(n_epoch))

                if not self.debug:
                    wandb.log({'train_loss': avg_train_loss,
                               'train_auroc': auroc_train,
                               'train_auprc': auprc_train})


                if best_eicu_auprc < eicu_auprc_eval:
                    best_eicu_loss = eicu_avg_eval_loss
                    best_eicu_auroc = eicu_auroc_eval
                    best_eicu_auprc = eicu_auprc_eval
                    torch.save({'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': best_eicu_loss,
                                'auroc': best_eicu_auroc,
                                'auprc': best_eicu_auprc,
                                'epochs': n_epoch}, self.best_eicu_eval_path)

                    print('[eicu] Model parameter saved at epoch {}'.format(n_epoch))

                print('[Train]  loss: {:.3f},  auroc: {:.3f},   auprc: {:.3f}'.format(avg_train_loss, auroc_train, auprc_train))
                print('[mimic/Valid]  loss: {:.3f},  auroc: {:.3f},   auprc: {:.3f}'.format(mimic_avg_eval_loss, mimic_auroc_eval, mimic_auprc_eval))
                print('[eicu/Valid]  loss: {:.3f},  auroc: {:.3f},   auprc: {:.3f}'.format(eicu_avg_eval_loss, eicu_auroc_eval, eicu_auprc_eval))

                self.mimic_early_stopping(mimic_auprc_eval)
                self.eicu_early_stopping(eicu_auprc_eval)

                if self.mimic_early_stopping.early_stop and self.eicu_early_stopping.early_stop:
                    print('Early stopping')

                    if not self.debug:
                        torch.save({'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict(),
                                    'epochs': n_epoch}, self.final_path)

                    break

        if self.source_file != 'both':
            self.test()
        elif self.source_file == 'both':
            self.test_both()


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

                if self.BCE and self.target != 'dx_depth1_unique':
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

    def evaluation_both(self):
        self.model.eval()

        preds_eval = []
        truths_eval = []
        mimic_avg_eval_loss = 0.0

        with torch.no_grad():
            for iter, sample in enumerate(self.mimic_valid_dataloader):

                item_embed, item_target, seq_len = sample
                item_embed = item_embed.to(self.device)
                item_target = item_target.to(self.device)
                y_pred = self.model(item_embed, seq_len)

                if self.BCE and self.target != 'dx_depth1_unique':
                    loss = self.criterion(y_pred, item_target.unsqueeze(1).float().to(self.device))
                else:
                    loss = self.criterion(y_pred, item_target.float().to(self.device))
                mimic_avg_eval_loss += loss.item() / len(self.mimic_valid_dataloader)

                probs_eval = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_eval += list(probs_eval.flatten())
                truths_eval += list(item_target.detach().cpu().numpy().flatten())

            mimic_auroc_eval = roc_auc_score(truths_eval, preds_eval)
            mimic_auprc_eval = average_precision_score(truths_eval, preds_eval, average='micro')

            if not self.debug:
                wandb.log({'mimic_eval_loss': mimic_avg_eval_loss,
                           'mimic_eval_auroc': mimic_auroc_eval,
                           'mimic_eval_auprc': mimic_auprc_eval})

        preds_eval = []
        truths_eval = []
        eicu_avg_eval_loss = 0.0

        with torch.no_grad():
            for iter, sample in enumerate(self.eicu_valid_dataloader):
                item_embed, item_target, seq_len = sample
                item_embed = item_embed.to(self.device)
                item_target = item_target.to(self.device)
                y_pred = self.model(item_embed, seq_len)

                if self.BCE and self.target != 'dx_depth1_unique':
                    loss = self.criterion(y_pred, item_target.unsqueeze(1).float().to(self.device))
                else:
                    loss = self.criterion(y_pred, item_target.float().to(self.device))
                eicu_avg_eval_loss += loss.item() / len(self.eicu_valid_dataloader)

                probs_eval = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_eval += list(probs_eval.flatten())
                truths_eval += list(item_target.detach().cpu().numpy().flatten())

            eicu_auroc_eval = roc_auc_score(truths_eval, preds_eval)
            eicu_auprc_eval = average_precision_score(truths_eval, preds_eval, average='micro')

            if not self.debug:
                wandb.log({'eicu_eval_loss': eicu_avg_eval_loss,
                           'eicu_eval_auroc': eicu_auroc_eval,
                           'eicu_eval_auprc': eicu_auprc_eval})

        return mimic_avg_eval_loss, mimic_auroc_eval, mimic_auprc_eval, eicu_avg_eval_loss, eicu_auroc_eval, eicu_auprc_eval

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
                if self.BCE and self.target != 'dx_depth1_unique':
                    loss = self.criterion(y_pred, item_target.unsqueeze(1).float().to(self.device))
                else:
                    loss = self.criterion(y_pred, item_target.float().to(self.device))
                avg_test_loss += loss.item() / len(self.test_dataloader)

                probs_test = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_test += list(probs_test.flatten())
                truths_test += list(item_target.detach().cpu().numpy().flatten())

            auroc_test = roc_auc_score(truths_test, preds_test)
            auprc_test = average_precision_score(truths_test, preds_test, average='micro')

            if not self.debug:
                wandb.log({'test_loss': avg_test_loss,
                           'test_auroc': auroc_test,
                           'test_auprc': auprc_test})

            print('[Test]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_test_loss, auroc_test, auprc_test))

    def test_both(self):
        ckpt = torch.load(self.best_mimic_eval_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        preds_test = []
        truths_test = []
        avg_test_loss = 0.

        with torch.no_grad():
            for iter, sample in enumerate(self.mimic_test_dataloader):

                item_embed, item_target, seq_len = sample
                item_embed = item_embed.to(self.device)
                item_target = item_target.to(self.device)
                y_pred = self.model(item_embed, seq_len)
                if self.BCE and self.target != 'dx_depth1_unique':
                    loss = self.criterion(y_pred, item_target.unsqueeze(1).float().to(self.device))
                else:
                    loss = self.criterion(y_pred, item_target.float().to(self.device))
                avg_test_loss += loss.item() / len(self.mimic_test_dataloader)

                probs_test = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_test += list(probs_test.flatten())
                truths_test += list(item_target.detach().cpu().numpy().flatten())

            auroc_test = roc_auc_score(truths_test, preds_test)
            auprc_test = average_precision_score(truths_test, preds_test, average='micro')

            if not self.debug:
                wandb.log({'mimic_test_loss': avg_test_loss,
                           'mimic_test_auroc': auroc_test,
                           'mimic_test_auprc': auprc_test})

            print('[Test/mimic]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_test_loss, auroc_test,
                                                                                            auprc_test))

        ckpt = torch.load(self.best_eicu_eval_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        preds_test = []
        truths_test = []
        avg_test_loss = 0.

        with torch.no_grad():
            for iter, sample in enumerate(self.eicu_test_dataloader):

                item_embed, item_target, seq_len = sample
                item_embed = item_embed.to(self.device)
                item_target = item_target.to(self.device)
                y_pred = self.model(item_embed, seq_len)

                if self.BCE and self.target != 'dx_depth1_unique':
                    loss = self.criterion(y_pred, item_target.unsqueeze(1).float().to(self.device))
                else:
                    loss = self.criterion(y_pred, item_target.float().to(self.device))
                avg_test_loss += loss.item() / len(self.eicu_test_dataloader)

                probs_test = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_test += list(probs_test.flatten())
                truths_test += list(item_target.detach().cpu().numpy().flatten())

            auroc_test = roc_auc_score(truths_test, preds_test)
            auprc_test = average_precision_score(truths_test, preds_test, average='micro')

            if not self.debug:
                wandb.log({'eicu_test_loss': avg_test_loss,
                           'eicu_test_auroc': auroc_test,
                           'eicu_test_auprc': auprc_test})

            print('[Test/eicu]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_test_loss, auroc_test, auprc_test))


