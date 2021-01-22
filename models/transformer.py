import torch
import torch.nn as nn

import pickle
import os
import math

"""
reference : https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""

class PositionalEncoding(nn.Module):
    def __init__(self, dimension, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.dimension = dimension
        # pe shape of (max_length, 1, dimension)



    def forward(self, x, lengths):
        # x, shape of (max_length, batch_size, dimension)
        # lengths, shape of (batch_size, max_length)
        B = x.size(1)   # batch_size
        lengths = lengths.transpose(0, 1)    # shape of (max_length, batch_size)

        with torch.no_grad():
            pe = torch.zeros(int(self.max_len), B, self.dimension)
            div_term = torch.exp(torch.arange(0, self.dimension, 2).float() * (-math.log(10000.0) / self.dimension))
            div_term = torch.cat(([div_term.unsqueeze(0)] * B), dim=0)

            i = range(B)
            pe[:, i, 0::2] = torch.sin(lengths[:, i].unsqueeze(1) * div_term[i])
            pe[:, i, 1::2] = torch.cos(lengths[:, i].unsqueeze(1) * div_term[i])

            x = x + pe

        output = self.dropout(x)    # shape of (max_len, batch_size, dimension)
        return output


class Transformer(nn.Module):
    def __init__(self, args, output_size, device, target_file, n_layer=2,attn_head=8, hidden_dim=256, vocab_size=None):
        super(Transformer, self).__init__()
        dropout = args.dropout
        self.bert_induced = args.bert_induced

        # bert_induced Transformer
        if args.bert_induced:
            self.hidden_dim = hidden_dim
            self.device = device
            self.cls_freeze = args.cls_freeze

            if args.concat:
                initial_embed_weight = pickle.load(open(os.path.join(args.input_path + 'embed_vocab_file', args.item,
                                                                     '{}_{}_{}_{}_concat_cls_initialized.pkl'.format(target_file, args.item, args.time_window, args.bert_model)), 'rb'))
            elif not args.concat:
                initial_embed_weight = pickle.load(open(os.path.join(args.input_path + 'embed_vocab_file', args.item,
                                                                     '{}_{}_{}_{}_cls_initialized.pkl'.format(target_file, args.item, args.time_window, args.bert_model)), 'rb'))
            initial_embed_weight = initial_embed_weight[1:, :]
            initial_embed_weight = torch.cat((torch.zeros(1, initial_embed_weight.size(1)), torch.randn(1, initial_embed_weight.size(1)), initial_embed_weight), dim=0)

            self.embed = nn.Embedding(initial_embed_weight.size(0), initial_embed_weight.size(1), _weight=initial_embed_weight, padding_idx=0)
            self.compress_fc = nn.Linear(initial_embed_weight.size(1), args.embedding_dim)

            self.pos_encoder = PositionalEncoding(args.embedding_dim, dropout, args.max_length)
            encoder_layers = nn.TransformerEncoderLayer(initial_embed_weight.size(1), attn_head, hidden_dim, dropout)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layer)

            self.output_fc = nn.Linear(args.embedding_dim, output_size)

        # singleTransformer
        elif not args.bert_induced:
            pass


    def forward(self, x, lengths):
        # lengths, shape of (batch_size, max_length)
        src_key_padding_mask = ((x==0) * 1)
        x = x.long()

        if self.bert_induced:
            if self.cls_freeze:
                with torch.no_grad():
                    x = self.embed(x).to(self.device)
            elif not self.cls_freeze:
                x = self.embed(x).to(self.device)
            x = self.compress_fc(x)
            x = x.permute(1, 0, 2)  # x, shape of (seq_len, batch_size, dimension)
            x = self.pos_encoder(x, lengths)

            output = self.transformer_encoder(src=x, src_key_padding_mask=src_key_padding_mask)



        # don't forget to transpose to (seq_len, batch_size, dimension)


        # src_key_padding_mask
