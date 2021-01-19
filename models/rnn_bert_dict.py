import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import pickle
import os

class dict_post_RNN(nn.Module):
    def __init__(self, args, output_size, device, target_file, n_layers=1):
        super().__init__()
        dropout = args.dropout
        self.bidirection = args.rnn_bidirection
        num_directions = 2 if self.bidirection else 1
        self.hidden_dim = args.hidden_dim
        self.device = device
        self.cls_freeze = args.cls_freeze

        if args.concat:
            initial_embed_weight = pickle.load(open(os.path.join('/home/jylee/data/pretrained_ehr/input_data/embed_vocab_file', args.item,
                                                             '{}_{}_{}_{}_concat_cls_initialized.pkl'.format(target_file, args.item, args.time_window, args.bert_model)), 'rb'))
        elif not args.concat:
            initial_embed_weight = pickle.load(open(os.path.join('/home/jylee/data/pretrained_ehr/input_data/embed_vocab_file', args.item,
                                                             '{}_{}_{}_{}_cls_initialized.pkl'.format(target_file, args.item, args.time_window, args.bert_model)), 'rb'))

        self.embed = nn.Embedding(initial_embed_weight.size(0), initial_embed_weight.size(1), _weight=initial_embed_weight)

        self.compress_fc = nn.Linear(initial_embed_weight.size(1), args.embedding_dim)

        if args.rnn_model_type == 'gru':
            self.model = nn.GRU(args.embedding_dim, self.hidden_dim, dropout=dropout, batch_first=True, bidirectional=self.bidirection)
        elif args.rnn_model_type == 'lstm':
            self.model = nn.LSTM(args.embedding_dim, self.hidden_dim, dropout=dropout, batch_first=True, bidirectional=self.bidirection)

        if self.bidirection:
            self.linear_1 = nn.Linear(num_directions * self.hidden_dim, self.hidden_dim)

        self.output_fc = nn.Linear(self.hidden_dim, output_size)

    def forward(self, x, lengths):
        B = x.size(0)
        lengths = lengths.squeeze(-1).long()
        x = x.long()

        if self.cls_freeze:
            with torch.no_grad():
                embedded = self.embed(x).to(self.device)

        elif not self.cls_freeze:
            embedded = self.embed(x).to(self.device)
        embedded = self.compress_fc(embedded)

        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.model(packed)
        output_seq, output_len = pad_packed_sequence(output, batch_first=True)

        i = range(B)

        # if not self.bidirection:
        output = output_seq[i, lengths - 1, :]
        # else:
        #     forward_output = output_seq[i, lengths - 1, :self.hidden_dim]
        #     backward_output = output_seq[i, 0, self.hidden_dim:]
        #     output = torch.cat((forward_output, backward_output), dim=-1)
        #     output = self.linear_1(output)

        output = self.output_fc(output)
        return output



