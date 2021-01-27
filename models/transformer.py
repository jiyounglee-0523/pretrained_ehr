import torch
import torch.nn as nn

import pickle
import os
import math

"""
reference : https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dimension, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        max_len = int(max_len)

        pe = torch.zeros(max_len + 1, embedding_dimension)   # +1 for cls and pad positional embedding
        position = torch.arange(0, max_len + 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dimension, 2).float() * (-math.log(10000.0) / embedding_dimension))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe shape of (max_length, dimension)

        self.positional_embed = nn.Embedding(int(max_len)+1, embedding_dimension, _weight=pe)     # padding_index=0을 해줘야 하나?

    def forward(self, x, offset_order):
        # x, shape of (max_length, batch_size, dimension)
        # offset_order, shape of (batch_size, max_length)
        offset_order = offset_order.transpose(0, 1).long()    # shape of (max_length, batch_size)

        with torch.no_grad():
            positional_embed = self.positional_embed(offset_order)
        x = x + positional_embed

        output = self.dropout(x)    # shape of (max_len, batch_size, dimension)
        return output


class Transformer(nn.Module):
    def __init__(self, args, output_size, device, target_file, n_layer=2,attn_head=8, hidden_dim=256, vocab_size=None):
        super(Transformer, self).__init__()
        dropout = args.dropout
        self.bert_induced = args.bert_induced
        self.device = device

        # bert_induced Transformer
        if args.bert_induced:
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
            #self.compress_fc = nn.Linear(initial_embed_weight.size(1), args.embedding_dim)
            embedding_dimension = initial_embed_weight.size(1)

        # singleTransformer
        elif not args.bert_induced:
            self.embed = nn.Embedding(vocab_size + 1, args.embedding_dim, padding_idx=0)
            embedding_dimension = args.embedding_dim

        self.pos_encoder = PositionalEncoding(embedding_dimension, dropout, args.max_length).to(device)
        encoder_layers = nn.TransformerEncoderLayer(embedding_dimension, attn_head, hidden_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layer)

        self.output_fc = nn.Linear(embedding_dimension, output_size)



    def forward(self, x, offset_order):
        # offset_order, shape of (batch_size, max_length)
        src_key_padding_mask = ((x==0) * 1).clone().detach().bool().to(self.device)
        x = x.long().to(self.device)
        offset_order = offset_order.to(x.device)

        if self.bert_induced:
            if self.cls_freeze:
                with torch.no_grad():
                    x = self.embed(x).to(self.device)
            elif not self.cls_freeze:
                x = self.embed(x).to(self.device)
            #x = self.compress_fc(x)

        elif not self.bert_induced:
            x = self.embed(x).to(self.device)

        x = x.permute(1, 0, 2)  # x, shape of (seq_len, batch_size, dimension)
        x = self.pos_encoder(x, offset_order)

        output = self.transformer_encoder(src=x,
                                          src_key_padding_mask=src_key_padding_mask)  # output shape of (max_len, batch_size, embedding)
        output = output[0]  # only use cls output, shape of (batch_size, embedding)
        output = self.output_fc(output)

        return output


