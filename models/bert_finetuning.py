import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import transformers
from transformers import AutoTokenizer, AutoModel

import math


class ClinicalBERT(nn.Module):
    def __init__(self, word_max_length):
        super().__init__()
        self.word_max_length = word_max_length
        self.model = AutoModel.from_pretrained("prajjwal1/bert-tiny")

    def forward(self, x):
        # reshape (B, S, W) -> (B*S, W)
        x['input_ids'] = x['input_ids'].reshape(-1, self.word_max_length).cuda()
        x['token_type_ids'] = x['token_type_ids'].reshape(-1, self.word_max_length).cuda()
        x['attention_mask'] = x['attention_mask'].reshape(-1, self.word_max_length).cuda()
        # print('pre-BERT data loaded DONE!')
        # print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')

        cls_output= self.model(**x)   # cls_output shape (B * S, 768)
        output = cls_output[1]
        # print('pre-BERT forward calculation DOEN!')
        # print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')

        return output


class post_predictivelayer(nn.Module):
    def __init__(self, args, output_size, n_layers=1):
        super().__init__()
        bert_model = args.bert_model

        if bert_model == 'bio_clinical_bert':
            self.prebert = ClinicalBERT(args.word_max_length)

        elif args.bert_freeze == False:
            for param in self.prebert.parameters():
                param.requires_grad = True

        self.max_length = int(args.max_length)
        dropout = args.dropout
        self.freeze = args.bert_freeze
        self.hidden_dim = args.hidden_dim

        self.embed_fc = nn.Linear(768, args.embedding_dim)     # hard_coding

        self.model = nn.GRU(args.embedding_dim, self.hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True, bidirectional=self.bidirection)

        self.output_fc = nn.Linear(self.hidden_dim, output_size)

    def forward(self, x, lengths):
        # y
        # goes through prebert

        self.model.flatten_parameters()

        if self.freeze:
            with torch.no_grad():
                x = self.prebert(x)

        else:
            x = self.prebert(x)

        x = x.reshape(-1, 150, 768)     # hard coding!
        lengths = lengths.squeeze(1).long().cpu()
        B = x.size(0)

        x = self.embed_fc(x)    # B, S, embedding_dim
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.model(packed)
        output_seq, output_len = pad_packed_sequence(output, batch_first=True)

        i = range(B)

        # if not self.bidirection:
        output = output_seq[i, lengths -1, :]
        output = self.output_fc(output)
        return output

