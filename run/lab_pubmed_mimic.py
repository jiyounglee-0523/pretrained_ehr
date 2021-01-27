import subprocess
import os

# Configuration before run

PATH = '/home/jylee/pretrained_ehr/rnn_model/'
SRC_PATH = PATH+'main.py'

device = 2
os.environ['CUDA_VISIBLE_DEVICES'] = str(device)


target_list = ['readmission', 'mortality', 'los>3day', 'los>7day', 'dx_depth1_unique']

for target in target_list:
   TRAINING_CONFIG = {
       "bert_induced": True,
       "source_file": 'both',
       "item": 'all',
       "max_length": 300,
       "bert_freeze": True,
       "target": target,
       "bert_model": 'bert',
       "device_number": device,
       "only_BCE": True,
       "transformer": True,
       "cls_freeze": True
   }

   TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

   # Run script
   subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)


