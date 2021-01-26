import subprocess
import os

# Configuration before run

PATH = '/home/jylee/pretrained_ehr/rnn_model/'
SRC_PATH = PATH+'main.py'

device = 5
os.environ['CUDA_VISIBLE_DEVICES'] = str(device)


target_list = ['los>7day']

for target in target_list:
   TRAINING_CONFIG = {
       "bert_induced": True,
       "source_file": 'both',
       "item": 'lab',
       "max_length": 150,
       "bert_freeze": True,
       "target": target,
       "bert_model": 'bert',
       "device_number": device,
       #"concat": True,
       "cls_freeze": True
   }

   TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

   # Run script
   subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)
