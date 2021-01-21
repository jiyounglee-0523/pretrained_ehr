import subprocess
import os

# Configuration before run

PATH = '/home/jylee/pretrained_ehr/rnn_model/'
SRC_PATH = PATH+'main.py'

device = 2

os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

TRAINING_CONFIG = {
    "bert_induced": True,
    "source_file": 'both',
    "target": 'dx_depth1_unique',
    "item": 'all',
    "max_length": 300,
    "bert_freeze": True,
    "device_number": device,
    "debug": True,
    "bert_model": 'bio_bert',
}

TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

# Run script
subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)
