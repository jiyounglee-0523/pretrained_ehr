import subprocess
import os

# Configuration before run

PATH = '/home/jylee/pretrained_ehr/rnn_model/'
SRC_PATH = PATH+'main.py'

device = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(device)


target_list = ['mortality']

for target in target_list:
   TRAINING_CONFIG = {
       "bert_induced": True,
       "source_file": 'both',
       "item": 'all',
       "batch_size": 512,
       "bert_freeze": True,
       "target": target,
       "bert_model": 'bert_small',
       "device_number": device,
       "lr": 1e-4,
       "transformer": True,
       "only_BCE": True,
       "wandb_project_name": 'check_los',
   }

   TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

   # Run script
   subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)
