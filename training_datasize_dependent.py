import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.multiprocessing as mp

from sklearn.metrics import roc_auc_score, average_precision_score

import os
import argparse
import pickle
import random
import wandb
import tqdm







def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_induced', action='store_true')
    parser.add_argument('--source_file', choices=['mimic', 'eicu', 'both'], type=str)
    parser.add_argument('--training_dataset_ratio', choices=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0], type=float)
    parser.add_argument('--target', choices=['readmission', 'mortality', 'los>3day', 'los>7day', 'dx_depth1_unique'], type=str)
    parser.add_argument('--item', choices=['lab', 'med', 'inf'], type=str)





if __name__ == '__main__':
    main()