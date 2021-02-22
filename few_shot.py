import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.multiprocessing as mp

from sklearn.metrics import roc_auc_score, average_precision_score

import os
import argparse
import pickle
import random
import tqdm

from models.DescEmb import *
from models.CodeEmb import *
from models.bert_finetuning import *
from utils.loss import *
from utils.trainer_utils import *

"""
few-shot: 0.0(zero-shot), 0.1, 0.3, 0.5, 0.7, 0.9, 1.0(full-shot = transfer learning)
"""


def get_test_dataloader(args, data_type='train', data_name=None):       # validation? test?
    if data_type == 'train':
        train_data = Few_Shot_Dataset(args, data_type=data_type)
        dataloader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=16)

    elif data_type == 'eval':
        eval_data = Few_Shot_Dataset(args, data_type=data_type, data_name=data_name)
        dataloader = DataLoader(dataset=eval_data, batch_size=args.batch_size, shuffle=True, num_workers=16)

    elif data_type == 'test':
        test_data = Few_Shot_Dataset(args, data_type=data_type, data_name=data_name)
        dataloader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, num_workers=16)

    return dataloader


class Few_Shot_Dataset(Dataset):
    def __init__(self, args, data_type, data_name=None):
        if data_name is None:
            test_file = args.test_file
        else:
            test_file = data_name
        self.target = args.target
        item = args.item
        self.max_length = args.max_length
        time_window = args.time_window
        self.word_max_length = args.word_max_length
        self.bert_induced = args.bert_induced
        source_file = args.source_file
        self.transformer = args.transformer
        few_shot = args.few_shot

        if test_file == 'both':
            if few_shot ==  0.0 or few_shot == 1.0:
                mimic_path = os.path.join(args.input_path[:-1], item,
                                          'mimic_{}_{}_{}_{}.pkl'.format(time_window, item, self.max_length, args.seed))
                eicu_path = os.path.join(args.input_path[:-1], item,
                                         'eicu_{}_{}_{}_{}.pkl'.format(time_window, item, self.max_length, args.seed))
            else:
                mimic_path = os.path.join(args.input_path[:-1], item,
                                          'mimic_{}_{}_{}_{}_{}.pkl'.format(time_window, item, self.max_length,
                                                                         args.seed, int(few_shot * 100)))
                eicu_path = os.path.join(args.input_path[:-1], item,
                                         'eicu_{}_{}_{}_{}_{}.pkl'.format(time_window, item, self.max_length,
                                                                       args.seed, int(few_shot * 100)))

            mimic = pickle.load(open(mimic_path, 'rb'))
            eicu = pickle.load(open(eicu_path, 'rb'))

            mimic = mimic.rename({'HADM_ID': 'ID'}, axis='columns')
            eicu = eicu.rename({'patientunitstayid': 'ID'}, axis='columns')

            mimic_item_name, mimic_item_target, mimic_item_offset_order = self.preprocess(mimic, data_type, item, time_window, self.target)
            eicu_item_name, eicu_item_target, eicu_item_offset_order = self.preprocess(eicu, data_type, item, time_window, self.target)

            mimic_item_name.extend(eicu_item_name)
            self.item_name = mimic_item_name

            mimic_item_offset_order = list(mimic_item_offset_order)
            eicu_item_offset_order = list(eicu_item_offset_order)
            mimic_item_offset_order.extend(eicu_item_offset_order)
            self.item_offset_order = pad_sequence(mimic_item_offset_order, batch_first=True)

            if self.target == 'dx_depth1_unique':
                mimic_item_target.extend(eicu_item_target)
                self.item_target = mimic_item_target
            else:
                self.item_target = torch.cat((mimic_item_target, eicu_item_target))

        else:
            if few_shot == 0.0 or few_shot == 1.0:
                path = os.path.join(args.input_path[:-1], item, '{}_{}_{}_{}_{}.pkl'.format(test_file, time_window,
                                                                                                 item, self.max_length,
                                                                                                 args.seed))
            else:
                path = os.path.join(args.input_path[:-1], item, '{}_{}_{}_{}_{}_{}.pkl'.format(test_file, time_window, item, self.max_length, args.seed, int(few_shot * 100)))
            data = pickle.load(open(path, 'rb'))

            # change column name
            if test_file == 'mimic':
                data = data.rename({'HADM_ID': 'ID'}, axis='columns')

            elif test_file == 'eicu':
                data = data.rename({'patientunitstayid': 'ID'}, axis='columns')

            self.item_name, self.item_target, self.item_offset_order = self.preprocess(data, data_type, item, time_window, self.target)

        if source_file == 'both':
            vocab_path = os.path.join(args.input_path + 'embed_vocab_file', item, 'both_{}_{}_{}_word2embed.pkl'.format(item, time_window, args.bert_model))
        else:
            vocab_path = os.path.join(args.input_path + 'embed_vocab_file', item, '{}_{}_{}_{}_word2embed.pkl'.format(test_file, item, time_window, args.bert_model))
        self.id_dict = pickle.load(open(vocab_path, 'rb'))

    def __len__(self):
        return len(self.item_name)

    def __getitem__(self, item):
        single_target = self.item_target[item]
        if self.target == 'dx_depth1_unique':
            single_target = [int(j) for j in single_target]
            target_list = torch.Tensor(single_target).long()

            single_target = torch.zeros(18)
            single_target[target_list - 1] = 1   # shape of 18

        # bert_induced
        single_item_name = self.item_name[item]
        seq_len = torch.Tensor([len(single_item_name)])
        embedding = []

        def embed_dict(x):
            return self.id_dict[x]
        embedding = list(map(embed_dict, single_item_name))  # list with length seq_len
        embedding = torch.Tensor(embedding)

        padding = torch.zeros(int(self.max_length) - embedding.size(0))
        embedding = torch.cat((embedding, padding), dim=-1)
        return embedding, single_target, seq_len


    def preprocess(self, cohort, data_type, item, time_window, target):
        if time_window == 'Total':
            name_window = '{}_name'.format(item)
            offset_window = 'order_offset'
            offset_order_window = '{}_offset_order'.format(item)
            id_window = '{}_id_{}hr'.format(item, time_window)
            target_fold = '{}_fold'.format(target)

        else:
            name_window = '{}_name_{}hr'.format(item, time_window)
            offset_window = 'order_offset_{}hr'.format(time_window)
            offset_order_window = '{}_offset_order_{}hr'.format(item, time_window)
            id_window = '{}_id_{}hr'.format(item, time_window)
            target_fold = '{}_fold'.format(target)
            if target == 'dx_depth1_unique':
                target_fold = 'dx_fold'

        # extract cohort
        cohort = cohort[['ID', name_window, offset_window, offset_order_window, id_window, target, target_fold]]
        cohort = cohort[cohort[target_fold] != -1]   # -1 is for unsampled

        if data_type == 'train':
            cohort = cohort[cohort[target_fold] == 1]

        elif data_type == 'eval':
            cohort = cohort[cohort[target_fold] == 2]

        elif data_type == 'test':
            cohort = cohort[cohort[target_fold] == 0]

        # drop with null item
        cohort = cohort[cohort.astype(str)[name_window] != '[]']

        # pad
        item_name = cohort[name_window].values.tolist()

        item_id = cohort[id_window].apply(lambda x: torch.Tensor(x)).values.tolist()
        item_id = pad_sequence(item_id, batch_first=True)

        item_offset_order = cohort[offset_order_window].apply(lambda x: torch.Tensor(x)).values.tolist()
        item_offset_order = pad_sequence(item_offset_order, batch_first=True)

        # target
        if target == 'dx_depth1_unique':
            item_target = cohort[target].values.tolist()

        else:
            item_target = torch.LongTensor(cohort[target].values.tolist()) # shape of (B)

        return item_name, item_target, item_offset_order


################################################################################


class Tester(nn.Module):
    def __init__(self, args, train_dataloader, valid_dataloader, test_dataloader, device, seed):
        super().__init__()
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.target = args.target

        lr = args.lr
        self.n_epochs = args.n_epochs

        self.criterion = nn.BCEWithLogitsLoss()
        if args.target == 'dx_depth1_unique':
            output_size = 18
        else:
            output_size = 1

        file_target_name = args.target
        if file_target_name == 'los>3day':
            file_target_name = 'los_3days'
        elif file_target_name == 'los>7day':
            file_target_name = 'los_7days'

        if args.DescEmb and args.bert_freeze and not args.cls_freeze:
            model_directory = 'cls_learnable'
            if args.source_file == 'both':
                self.model = DescEmb(args=args, output_size=output_size, device=self.device, target_file='both').to(device)
            else:
                self.model = DescEmb(args=args, output_size=output_size, device=self.device, target_file=args.test_file).to(device)
            print('bert freeze, cls_learnable')

            filename = 'cls_learnable_{}_{}'.format(args.bert_model, args.seed)


        elif args.bert_induced and args.bert_freeze and args.cls_freeze:
            model_directory = 'cls_learnable'
            if args.source_file == 'both':
                self.model = DescEmb(args=args, output_size=output_size, device=self.device, target_file='both').to(device)
            else:
                self.model = DescEmb(args=args, output_size=output_size, device=self.device, target_file=args.test_file).to(device)
            print('bert freeze, cls_freeze, RNN')
            filename = 'cls_fixed_{}_{}'.format(args.bert_model, args.seed)

        elif args.bert_induced and not args.bert_freeze and not args.transformer:
            model_directory = 'bert_finetune'
            self.model = post_predictivelayer(args=args, output_size=output_size).to(device)
            print('bert finetuning')


        elif not args.bert_induced:
            model_directory = 'singleRNN'

            if args.source_file == 'both':
                if args.item == 'lab':
                    vocab_size = 14371 if args.concat else 448
                elif args.item == 'med':
                    vocab_size = 4898 if args.concat else 2812
                elif args.item == 'inf':
                    vocab_size = 979
                elif args.item == 'all':
                    vocab_size = 15794 if args.concat else 3672
            else:
                if args.test_file == 'mimic':
                    if args.item == 'lab':
                        vocab_size = 5110 if args.concat else 359
                    elif args.item == 'med':
                        vocab_size = 2211 if args.concat else 1535
                    elif args.item == 'inf':
                        vocab_size = 485
                    elif args.item == 'all':
                        vocab_size = 7563 if args.concat else 2377
                elif args.test_file == 'eicu':
                    if args.item == 'lab':
                        vocab_size = 9659 if args.concat else 134
                    elif args.item == 'med':
                        vocab_size = 2693 if args.concat else 1283
                    elif args.item == 'inf':
                        vocab_size = 495
                    elif args.item == 'all':
                        vocab_size = 8532 if args.concat else 1344
            self.model = CodeEmb(args, vocab_size, output_size, self.device).to(device)
            print('single rnn')
            filename = 'trained_single_rnn_{}'.format(args.seed)


        self.source_path = os.path.join(args.path, args.item, model_directory, args.source_file, file_target_name, filename)

        if args.cls_freeze:
            target_filename = 'few_shot{}_from{}_to{}_model{}_seed{}_clsfixed_onlyBCE'.format(args.few_shot, args.source_file,
                args.test_file, args.bert_model, seed)

        elif not args.cls_freeze:
            target_filename = 'few_shot{}_from{}_to{}_model{}_seed{}_onlyBCE'.format(args.few_shot, args.source_file, args.test_file,
                                                                                     args.bert_model, seed)

        target_path = os.path.join(args.path, args.item, model_directory, args.test_file, file_target_name, target_filename)

        self.best_target_path = target_path + '_best_auprc.pt'
        self.final_path = target_path + '_final.pt'

        # load parameters
        best_eval_path = self.source_path + '_best_auprc.pt'
        print('Load Model from {}'.format(best_eval_path))
        ckpt = torch.load(best_eval_path)
        if args.source_file == 'both':
            self.model.load_state_dict(ckpt['model_state_dict'])
            print("Model fully loaded!")
        else:
            if args.source_file != args.test_file:
                pretrained_dict = ckpt['model_state_dict']
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if not 'embedding' in k}    # do not load embedding weight (singleRNN)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if not 'embed' in k}     # do not load embedding weight (bert-induced)
                self.model.load_state_dict(pretrained_dict, strict=False)
                print("Model partially loaded!")

            elif args.source_file == args.test_file:
                self.model.load_state_dict(ckpt['model_state_dict'])
                print("Model fully loaded!")

        print('Model will be saved in {}'.format(self.best_target_path))

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
            avg_train_loss = 0.

            for iter, sample in tqdm.tqdm(enumerate(self.train_dataloader)):
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                item_id, item_target, seq_len = sample
                item_id = item_id.to(self.device)
                item_target = item_target.to(self.device)

                y_pred = self.model(item_id, seq_len)
                if self.target != 'dx_depth1_unique':
                    loss = self.criterion(y_pred, item_target.unsqueeze(1).float().to(self.device))
                else:
                    loss = self.criterion(y_pred, item_target.float().to(self.device))

                loss.backward()
                self.optimizer.step()

                avg_train_loss += loss.item() / len(self.train_dataloader)

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
                            'epochs': n_epoch}, self.best_target_path)
                print('Model parameter saved at epoch {}'.format(n_epoch))

            print('[Train]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_train_loss, auroc_train, auprc_train))
            print('[Valid]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_eval_loss, auroc_eval, auprc_eval))

            self.early_stopping(auprc_eval)
            if self.early_stopping.early_stop:
                print('Early stopping')
                self.test()
                break

    def evaluation(self):
        self.model.eval()

        preds_eval = []
        truths_eval = []
        avg_eval_loss = 0.

        with torch.no_grad():
            for iter, sample in enumerate(self.valid_dataloader):
                item_id, item_target, seq_len = sample
                item_id = item_id.to(self.device)
                item_target = item_target.to(self.device)

                y_pred = self.model(item_id, seq_len)
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


    def zero_shot_test(self):
        self.model.eval()

        preds_test = []
        truths_test = []
        avg_test_loss = 0.

        with torch.no_grad():
            for iter, sample in enumerate(self.test_dataloader):
                item_id, item_target, seq_len = sample
                item_id = item_id.to(self.device)
                item_target = item_target.to(self.device)

                y_pred = self.model(item_id, seq_len)
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

            print('[Test]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_test_loss, auroc_test,
                                                                                     auprc_test))


    def test(self):
        ckpt = torch.load(self.best_target_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        preds_test = []
        truths_test = []
        avg_test_loss = 0.

        with torch.no_grad():
            for iter, sample in enumerate(self.test_dataloader):
                item_id, item_target, seq_len = sample
                item_id = item_id.to(self.device)
                item_target = item_target.to(self.device)

                y_pred = self.model(item_id, seq_len)
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

            print('[Test]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_test_loss, auroc_test,
                                                                                     auprc_test))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--DescEmb', action='store_true')
    parser.add_argument('--source_file', choices=['mimic', 'eicu'], type=str, default='mimic')
    parser.add_argument('--test_file', choices=['mimic', 'eicu', 'both'], type=str, default='eicu')
    parser.add_argument('--few_shot', type=float, choices=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], default=0.0)
    parser.add_argument('--target', choices=['readmission', 'mortality', 'los>3day', 'los>7day', 'dx_depth1_unique'], type=str, default='readmission')
    parser.add_argument('--item', choices=['lab', 'med', 'inf', 'all'], type=str, default='med')
    parser.add_argument('--time_window', choices=['12', '24', '36', '48', 'Total'], type=str, default='12')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_length', type=str, default='150')
    parser.add_argument('--bert_model', choices=['bio_clinical_bert', 'bio_bert', 'pubmed_bert', 'blue_bert', 'bert_mini', 'bert_tiny'], type=str, default='bio_bert')
    parser.add_argument('--bert_freeze', action='store_true')
    parser.add_argument('--input_path', type=str, default='/home/jylee/data/pretrained_ehr/input_data/', help='data directory')
    parser.add_argument('--path', type=str, default='/home/jylee/data/pretrained_ehr/output/KDD_output/', help='model parameter directory')
    parser.add_argument('--word_max_length', type=int, default=15)  # tokenized word max_length, used in padding
    parser.add_argument('--device_number', type=str, default='7')
    parser.add_argument('--cls_freeze', action='store_true')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_number)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.source_file == args.test_file:
        assert args.few_shot == 0.0, "there is no few_shot if source and test file are the same"

    mp.set_sharing_strategy('file_system')

    SEED = [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029]

    for seed in SEED:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

        args.seed = seed

        train_loader = get_test_dataloader(args=args, data_type='train')
        valid_loader = get_test_dataloader(args=args, data_type='eval')
        test_loader = get_test_dataloader(args=args, data_type='test')

        tester = Tester(args, train_loader, valid_loader, test_loader, device, seed)

        if args.few_shot == 0.0:
            print('Only test')
            tester.zero_shot_test()
        else:
            print('Train then test')
            tester.train()

if __name__ == '__main__':
    main()

