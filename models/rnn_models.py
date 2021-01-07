import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

"""
To-Do : (max, avg) pooling 어떻게 할지? 
        n_layers, dropout 도 argument로 받을 것인지
"""



class RNNmodels(nn.Module):
    def __init__(self, args, vocab_size, output_size, device, n_layers = 1):
        super(RNNmodels, self).__init__()
        self.bidirection = bool(args.rnn_bidirection)
        embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        num_directions = 2 if self.bidirection else 1
        dropout = args.dropout

        self.device = device

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if args.rnn_model_type == 'gru':
            self.model = nn.GRU(embedding_dim, self.hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True, bidirectional=self.bidirection)
        elif args.rnn_model_type == 'lstm':
            self.model = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True, bidirectional=self.bidirection)

        if self.bidirection:
            self.linear_1 = nn.Linear(num_directions * self.hidden_dim, self.hidden_dim)

        self.output_fc = nn.Linear(self.hidden_dim, output_size)

        # self.linear_1 = nn.Linear(hidden_dim * num_directions, )     # linear1은 좀 더 생각하기

    def forward(self, x, lengths):
        x = self.embedding(x.long().to(self.device))

        lengths = lengths.squeeze(-1).long()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.model(packed)
        output_seq, output_len = pad_packed_sequence(output, batch_first=True)

        i = range(x.size(0))

        if not self.bidirection:    # unidirectional
            output = output_seq[i, lengths -1, :]
        else:
            forward_output = output_seq[i, lengths -1, :self.hidden_dim]
            backward_output = output_seq[:, 0, self.hidden_dim:]
            output = torch.cat((forward_output, backward_output), dim=-1)
            output = self.linear_1(output)

        output = self.output_fc(output)
        return output



        