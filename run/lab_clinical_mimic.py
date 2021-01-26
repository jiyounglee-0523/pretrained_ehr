import subprocess
import os

# Configuration before run

PATH = '/home/jylee/pretrained_ehr/rnn_model/'
SRC_PATH = PATH+'main.py'

device = 7
os.environ['CUDA_VISIBLE_DEVICES'] = str(device)


target_list = ['los>7day']
bert_model_list = ['blue_bert', 'bert']

for bert in bert_model_list:
   TRAINING_CONFIG = {
       "bert_induced": True,
       "source_file": 'both',
       "item": 'lab',
       "bert_freeze": True,
       "max_length": 150,
       "target": 'los>7day',
       "bert_model": bert,
       "device_number": device,
       "concat": True,
       "cls_freeze": True
   }

   TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

   # Run script
   subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)
