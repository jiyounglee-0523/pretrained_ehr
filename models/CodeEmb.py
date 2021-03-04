import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

"""
To-Do : (max, avg) pooling 어떻게 할지? 
        n_layers, dropout 도 argument로 받을 것인지
"""



class CodeEmb(nn.Module):
    def __init__(self, args, vocab_size, output_size, device, n_layers = 1):
        super(CodeEmb, self).__init__()
        embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        dropout = args.dropout

        self.device = device

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.model = nn.GRU(embedding_dim, self.hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True)

        self.output_fc = nn.Linear(self.hidden_dim, output_size)



    def forward(self, x, lengths):
        x = self.embedding(x.long().to(self.device))

        lengths = lengths.squeeze(-1).long()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.model(packed)
        output_seq, output_len = pad_packed_sequence(output, batch_first=True)

        i = range(x.size(0))

        output = output_seq[i, lengths -1, :]

        output = self.output_fc(output)
        return output



        