import os

import wandb
from sklearn.metrics import roc_auc_score, average_precision_score

from models.rnn_bert_dict import *
from models.rnn_models import *
from utils.loss import *
from utils.trainer_utils import *


class Tester(nn.Module):
    def __init__(self, args, test_dataloader, device):
        super().__init__()

        self.dataloader = test_dataloader
        self.device = device
        wandb.init(project='pretrained_ehr_team', config=args)
        args = wandb.config

        file_target_name = args.target
        if file_target_name == 'los>3day':
            file_target_name = 'los_3day'
        elif file_target_name == 'los>7day':
            file_target_name = 'los_7day'

        if args.bert_induced:
            self.bert_induced = 'bert_induced_True'
        else:
            self.bert_induced = 'bert_induced_False'

        filename = 'dropout{}_emb{}_hid{}_bidirect{}_lr{}'.format(args.dropout, args.embedding_dim, args.hidden_dim,
                                                                  args.rnn_bidirection, args.lr)
        print(os.path.join(args.path, self.bert_induced, args.source_file, file_target_name))
        self.path = os.path.join(args.path, self.bert_induced, args.source_file, file_target_name, filename)
        print('Model is loadded from {}'.format(self.path))

        if args.source_file == 'mimic':
            vocab_size = 600  ########### change this!   vocab size 잘못됨
        elif args.source_file == 'eicu':
            vocab_size = 150
        else:
            raise NotImplementedError

        if args.target == 'dx_depth1_unique':
            output_size = 18
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            output_size = 1
            self.criterion = FocalLoss()

        if args.bert_induced == True:
            self.model = dict_post_RNN(args=args, output_size=output_size).to(self.device)
        else:
            self.model = RNNmodels(args=args, vocab_size=vocab_size, output_size=output_size, device=device).to(self.device)

    def test(self):
        preds_test_list=[]
        for valid_index in range(5):
            valid_index += 1
            print('{}_fold test start'.format(valid_index))
            self.valid_index = '_fold' + str(valid_index)
            self.best_eval_path = self.path + self.valid_index + '_best_eval.pt'
            self.final_path = self.path + self.valid_index + '_final.pt'

            self.model.eval()

            if os.path.exists(self.best_eval_path):
                ckpt = torch.load(self.best_eval_path)
                self.model.load_state_dict(ckpt['model_state_dict'])
                best_loss = ckpt['loss']
                best_auroc = ckpt['auroc']
                best_auprc = ckpt['auprc']
            else:
                raise NotImplementedError

            preds_test = []
            truths_test = []
            avg_test_loss = 0.

            with torch.no_grad():
                for iter, sample in enumerate(self.dataloader):
                    item_id, item_offset, item_offset_order, lengths, target = sample
                    item_id.to(self.device)
                    item_offset.to(self.device)
                    item_offset_order.to(self.device)
                    lengths.to(self.device)
                    target.to(self.device)

                    y_pred = self.model(item_id, lengths)
                    loss = self.criterion(y_pred, target.float().to(self.device))
                    avg_test_loss += loss.item() / len(self.dataloader)

                    probs_test = torch.sigmoid(y_pred).detach().cpu().numpy()
                    preds_test += list(probs_test.flatten())
                    truths_test += list(target.detach().cpu().numpy().flatten())

                auroc_test = roc_auc_score(truths_test, preds_test)
                auprc_test = average_precision_score(truths_test, preds_test)
                preds_test_list.append(preds_test)

                wandb.log({'test_loss': avg_test_loss,
                           'test_auroc': auroc_test,
                           'test_auprc': auprc_test})

                print('[test_{}_fold]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(valid_index,avg_test_loss, auroc_test,
                                                                                            auprc_test))
                print('[best_valid_{}_fold]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(valid_index,best_loss, best_auroc,
                                                                                                     best_auprc))
        preds_avg = np.mean(np.array(preds_test_list), axis=0)
        auroc_test_avg = roc_auc_score(truths_test, preds_avg)
        auprc_test_avg = average_precision_score(truths_test, preds_avg)

        print('[test_avg]    auroc: {:.3f},     auprc:   {:.3f}'.format(auroc_test_avg, auprc_test_avg))

