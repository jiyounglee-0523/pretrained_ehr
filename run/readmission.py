import subprocess
import os

# Configuration before run

PATH = '/home/jylee/pretrained_ehr/rnn_model/'
SRC_PATH = PATH+'main3.py'

device = 7

os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

TRAINING_CONFIG = {
    #"bert_induced": True,
    "source_file": 'eicu',
    "target": 'dx_depth1_unique',
    "bert_freeze": True,
    "device_number": device
}

TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

# Run script
subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)
