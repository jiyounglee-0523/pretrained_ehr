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


class post_RNN(nn.Module):
    def __init__(self, args, output_size, n_layers=1):
        super().__init__()
        bert_model = args.bert_model

        if bert_model == 'bio_clinical_bert':
            self.prebert = ClinicalBERT(args.word_max_length)

        if args.bert_freeze == True:
            for param in self.prebert.parameters():
                param.requires_grad = False

        elif args.bert_freeze == False:
            for param in self.prebert.parameters():
                param.requires_grad = True

        self.max_length = int(args.max_length)
        dropout = args.dropout
        self.freeze = args.bert_freeze

        self.bidirection = args.rnn_bidirection
        num_directions = 2 if self.bidirection else 1
        self.hidden_dim = args.hidden_dim

        self.embed_fc = nn.Linear(768, args.embedding_dim)     # hard_coding

        if args.rnn_model_type == 'gru':
            self.model = nn.GRU(args.embedding_dim, self.hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True, bidirectional=self.bidirection)
        elif args.rnn_model_type == 'lstm':
            self.model = nn.LSTM(args.embedding_dim, self.hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True, bidirectional=self.bidirection)

        if self.bidirection:
            self.linear_1 = nn.Linear(num_directions * self.hidden_dim, self.hidden_dim)

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
        # else:
        #     forward_output = output_seq[i, lengths -1, :self.hidden_dim]
        #     backward_output = output_seq[i, 0, self.hidden_dim:]
        #     output = torch.cat((forward_output, backward_output), dim=-1)
        #     output = self.linear_1(output)

        output = self.output_fc(output)
        return output

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


class post_Transformer(nn.Module):
    def __init__(self, args, output_size, device, n_layers=2, attn_head=8, hidden_dim=256):
        super().__init__()
        dropout = args.dropout
        self.max_length = args.max_length
        self.device = device
        self.freeze = args.bert_freeze
        word_max_length = 40 if (args.item == 'med' and args.source_file=='eicu')  else args.word_max_length

        if args.bert_model == 'bert_tiny':
            self.prebert = ClinicalBERT(word_max_length)

        if args.bert_freeze == True:
            for param in self.prebert.parameters():
                param.requires_grad = False

        elif args.bert_freeze == False:
            for param in self.prebert.parameters():
                param.requires_grad = True

        self.max_length = int(args.max_length)
        self.hidden_dim = args.hidden_dim

        self.cls_embed = nn.Embedding(1, 128)
        self.pos_encoder = PositionalEncoding(128, dropout, args.max_length).to(device)
        encoder_layers = nn.TransformerEncoderLayer(128, attn_head, hidden_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)

        self.output_fc = nn.Linear(128, output_size)

    def forward(self, x, offset_order, src_key_padding_mask):
        #offset_order, shape of (batch_size, max_length)
        src_key_padding_mask = src_key_padding_mask.clone().detach().bool().to(self.device)
        #src_key_padding_mask = ((x['attention_mask']==0) * 1).clone().detach().bool().to(self.device)

        if self.freeze:
            with torch.no_grad():
                x = self.prebert(x)
        else:
            x = self.prebert(x)

        x = x.reshape(-1, self.max_length, 128)    # x, shape of (batch_size, seq_len, dimension)
        B = x.size(0)
        cls = torch.zeros(B, 1).long().to(self.device)
        cls = self.cls_embed(cls).to(self.device)   # batch_size, seq_len, dimension
        x = torch.cat((cls, x), dim=1)

        x = x.permute(1, 0, 2)    # x, shape of (seq_len, batch_size, dimension)
        x = self.pos_encoder(x, offset_order)

        output = self.transformer_encoder(src=x, src_key_padding_mask=src_key_padding_mask)
        output = output[0]
        output = self.output_fc(output)
        return output








