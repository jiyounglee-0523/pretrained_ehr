import torch
import torch.nn as nn

import pickle
import os

"""
reference : https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""

class PositionalEncoding(nn.Module):
    pass   ################ work on this!


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
            self.embed = nn.Embedding(initial_embed_weight.size(0), initial_embed_weight.size(1), _weight=initial_embed_weight)
            self.compress_fc = nn.Linear(initial_embed_weight.size(1), args.embedding_dim)

            self.pos_encoder = PositionalEncoding()
            encoder_layers = nn.TransformerEncoderLayer(initial_embed_weight.size(1), attn_head, hidden_dim, dropout)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layer)


            self.output_fc = nn.Linear(args.embedding_dim, output_size)

        # singleTransformer
        elif not args.bert_induced:
            pass



    def generate_attn_mask(self):
        pass

    def forward(self, x, lengths):
        if self.bert_induced:
            if self.cls_freeze:
                x = self.embed(x).to(self.device)
                x = self.pos_encoder(x)



        # src_key_padding_mask
