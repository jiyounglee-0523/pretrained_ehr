import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import transformers
from transformers import AutoTokenizer, AutoModel

class ClinicalBERT(nn.Module):
    def __init__(self, word_max_length):
        super().__init__()
        self.word_max_length = word_max_length
        self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    def forward(self, x):
        # reshape (B, S, W) -> (B*S, W)
        x['input_ids'] = x['input_ids'].reshape(-1, self.word_max_length).cuda()
        x['token_type_ids'] = x['token_type_ids'].reshape(-1, self.word_max_length).cuda()
        x['attention_mask'] = x['attention_mask'].reshape(-1, self.word_max_length).cuda()
        # print('pre-BERT data loaded DONE!')
        # print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')

        _, cls_output = self.model(**x)   # cls_output shape (B * S, 768)
        # print('pre-BERT forward calculation DOEN!')
        # print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')

        return cls_output


class post_RNN(nn.Module):
    def __init__(self, args, output_size, n_layers=1):
        super().__init__()
        bert_model = args.bert_model
        if bert_model == 'clinical_bert':
            self.prebert = ClinicalBERT(args.word_max_length)

        if args.bert_freeze:
            for param in self.prebert.parameters():
                param.requires_grad = False

        elif args.bert_freeze == False:
            for param in self.prebert.parameters():
                param.requires_grad = True

        self.max_length = int(args.max_length)
        dropout = args.dropout

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
        # goes through prebert
        x = self.prebert(x)

        # with torch.no_grad():
        #     x = self.prebert(x)

        x = x.reshape(-1, 150, 768)     # hard coding!
        lengths = lengths.squeeze().long()
        B = x.size(0)

        # if self.freeze:    # freeze the output
        #     x = x.detach()

        x = self.embed_fc(x)    # B, S, embedding_dim
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.model(packed)
        output_seq, output_len = pad_packed_sequence(output, batch_first=True)

        i = range(B)

        if not self.bidirection:
            output = output_seq[i, lengths -1, :]
        else:
            forward_output = output_seq[i, lengths -1, :self.hidden_dim]
            backward_output = output_seq[i, 0, self.hidden_dim:]
            output = torch.cat((forward_output, backward_output), dim=-1)
            output = self.linear_1(output)

        output = self.output_fc(output)
        return output

