import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, args, output_size, device, target_file, vocab_size=None):
        super(Transformer, self).__init__()
        # bert_induced Transformer
        if args.bert_induced:
            # bert_induced Transformer




        # singleTransformer
        elif not args.bert_induced: