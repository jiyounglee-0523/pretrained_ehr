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
import wandb
import tqdm

from models.rnn_bert_dict import *
from models.rnn_models import *
from utils.loss import *
from utils.trainer_utils import *


def get_data_loader(args, data_type):
    if data_type == 'train':
        train_data = PoolBase(args, data_type)
        dataloader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=16)
    elif data_type == 'eval':
        eval_data = PoolBase(args, data_type)
        dataloader = DataLoader(dataset=eval_data, batch_size=args.batch_size, shuffle=True, num_workers=16)
    elif data_type == 'test':
        test_data = PoolBase(args, data_type)
        dataloader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, num_workers=16)
    return dataloader


class PoolBase(Dataset):
    def __init__(self, args, data_type):
        self.target = args.target
        item = args.item
        self.max_length = args.max_length
        time_window = args.time_window
        self.word_max_length = args.word_max_length
        self.bert_induced = args.bert_induced
        base_file = args.base_file
        test_file = args.test_file
        few_shot = args.few_shot

        base_data = os.path.join(args.input_path[:-1], item, '{}_{}_{}_{}_{}.pkl'.format(base_file, time_window, item, self.max_length, args.seed))
        if few_shot == 1.0:
            test_data = os.path.join(args.input_path[:-1], item, '{}_{}_{}_{}_{}.pkl'.format(test_file, time_window, item, self.max_length, args.seed))
        else:
            test_data = os.path.join(args.input_path[:-1], item, '{}_{}_{}_{}_{}_{}.pkl'.format(test_file, time_window, item, self.max_length, args.seed, int(few_shot * 100)))

        base_data = pickle.load(open(base_data, 'rb'))
        test_data = pickle.load(open(test_data, 'rb'))

        base_item_name, base_item_target = self.preprocess(base_data, data_type, item, time_window, self.target)
        test_item_name, test_item_target = self.preprocess(test_data, data_type, item, time_window, self.target)

        if data_type == 'train':
            base_item_name.extend(test_item_name)
            self.item_name = base_item_name

            if self.target == 'dx_depth1_unique':
                base_item_target.extend(test_item_target)
                self.item_target = base_item_target
            else:
                self.item_target = torch.cat((base_item_target, test_item_target))

        else:
            self.item_name = test_item_name
            self.item_target = test_item_target

        vocab_path = os.path.join(args.input_path + 'embed_vocab_file', item, 'both_{}_{}_{}_word2embed.pkl'.format(item, time_window, args.bert_model))
        self.id_dict = pickle.load(open(vocab_path, 'rb'))

    def __len__(self):
        return len(self.item_name)

    def __getitem__(self, item):
        single_target = self.item_target[item]
        if self.target == 'dx_depth1_unique':
            single_target = [int(j) for j in single_target]
            target_list = torch.Tensor(single_target).long()

            single_target = torch.zeros(18)
            single_target[target_list - 1] = 1  # shape of 18

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
        name_window = '{}_name_{}hr'.format(item, time_window)
        offset_window = 'order_offset_{}hr'.format(time_window)
        offset_order_window = '{}_offset_order_{}hr'.format(item, time_window)
        id_window = '{}_id_{}hr'.format(item, time_window)
        target_fold = '{}_fold'.format(target)
        if target == 'dx_depth1_unique':
            target_fold = 'dx_fold'

        # extract cohort
        cohort = cohort[['ID', name_window, offset_window, offset_order_window, id_window, target, target_fold]]
        cohort = cohort[cohort[target_fold] != -1]  # -1 is for unsampled

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

        # item_offset_order = cohort[offset_order_window].apply(lambda x: torch.Tensor(x)).values.tolist()
        # item_offset_order = pad_sequence(item_offset_order, batch_first=True)
        #
        # item_offset = cohort[offset_window].apply(lambda x: torch.Tensor(x)).values.tolist()
        # item_offset = pad_sequence(item_offset, batch_first=True)

        # target
        if target == 'dx_depth1_unique':
            item_target = cohort[target].values.tolist()

        else:
            item_target = torch.LongTensor(cohort[target].values.tolist())  # shape of (B)

        return item_name, item_target


class PoolBase_Trainer():
    def __init__(self, args, device):
        self.train_dataloader = get_data_loader(args, data_type='train')
        self.valid_dataloader = get_data_loader(args, data_type='eval')
        self.test_dataloader = get_data_loader(args, data_type='test')

        self.device = device
        self.debug = args.debug

        if not self.debug:
            wandb.init(project=args.wandb_project_name, entity='pretrained_ehr', config=args, reinit=True)

        lr = args.lr
        self.n_epochs = args.n_epochs
        self.BCE = args.only_BCE
        self.target = args.target

        if not args.only_BCE:
            raise NotImplementedError
        else:
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

        if args.bert_induced:
            model_directory = 'cls_learnable'
            self.model = dict_post_RNN(args=args, output_size=output_size, device=self.device, target_file='both').to(device)
            print('bert freeze, cls_learnable, RNN')
            filename = 'cls_learnable_rnn_base{}_test{}_{}_{}_dataportion{}'.format(args.base_file, args.test_file, args.bert_model, args.seed, args.few_shot)

        else:
            model_directory = 'singleRNN'
            if args.item == 'lab':
                vocab_size = 14371 if args.concat else 448
            elif args.item == 'med':
                vocab_size = 4898 if args.concat else 2812
            elif args.item == 'inf':
                vocab_size = 979
            elif args.item == 'all':
                vocab_size = 15794 if args.concat else 3672

            self.model = RNNmodels(args, vocab_size, output_size, self.device).to(device)
            print('singleRNN')

            filename = 'trained_single_rnn_base{}_test{}_{}_dataportion{}'.format(args.base_file, args.test_file, args.seed, args.few_shot)

        path = os.path.join(args.path, args.item, model_directory, 'both', file_target_name, filename)
        print('Model will be saved in {}'.format(path))

        self.best_eval_path = path + '_best_auprc.pt'
        self.final_path = path + '_final.pt'

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

            print('[Train]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_train_loss, auroc_train,
                                                                                         auprc_train))
            print('[Valid]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_eval_loss, auroc_eval,
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
        self.test()


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


    def test(self):
        ckpt = torch.load(self.best_eval_path)
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

                print('[Test]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_test_loss, auroc_test,
                                                                                            auprc_test))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_induced', action='store_true')
    parser.add_argument('--base_file', choices=['mimic', 'eicu'])
    parser.add_argument('--test_file', choices=['mimic', 'eicu'])
    parser.add_argument('--few_shot', type=float, choices=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    parser.add_argument('--target', choices=['readmission', 'mortality', 'los>3day', 'los>7day', 'dx_depth1_unique'], type=str)
    parser.add_argument('--item', choices=['lab', 'med', 'inf', 'all'])
    parser.add_argument('--time_window', choices=['12', '24', '36', '48', 'Total'], type=str, default='12')
    parser.add_argument('--rnn_model_type', choices=['gru', 'lstm'], type=str, default='gru')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--rnn_bidirection', action='store_true')
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_length', type=str, default='150')
    parser.add_argument('--bert_model', choices=['bio_clinical_bert', 'bio_bert', 'pubmed_bert', 'blue_bert', 'bert_mini', 'bert_tiny'], type=str, default='bio_clinical_bert')
    parser.add_argument('--bert_freeze', action='store_true')
    parser.add_argument('--input_path', type=str, default='/home/jylee/data/pretrained_ehr/input_data/')
    parser.add_argument('--path', type=str, default='/home/jylee/data/pretrained_ehr/output/KDD_output/')
    parser.add_argument('--word_max_length', type=int, default=15)  # tokenized word max_length, used in padding
    parser.add_argument('--device_number', type=str, default='7')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--cls_freeze', action='store_true')
    parser.add_argument('--only_BCE', action='store_true')
    parser.add_argument('--concat', action='store_true')
    parser.add_argument('--wandb_project_name', type=str)
    args = parser.parse_args()

    assert args.base_file != args.test_file, 'base file and test file should be different'

    args.time_window = '12'
    args.rnn_model_type = 'gru'
    args.batch_size = 512
    args.rnn_bidirection = False
    args.n_epochs = 1000
    args.word_max_length = 15  # tokenized word max_length, used in padding
    # hyperparameter tuning
    args.dropout = 0.3
    args.embedding_dim = 128
    args.hidden_dim = 256
    args.lr = 1e-4

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_number)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        Trainer = PoolBase_Trainer(args, device)

        Trainer.train()



if __name__ == '__main__':
    main()