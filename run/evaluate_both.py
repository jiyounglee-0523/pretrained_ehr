import subprocess
import os

# Configuration before run

PATH = '/home/jylee/pretrained_ehr/rnn_model/'
SRC_PATH = PATH+'main.py'

device = 7


target_list = ['mortality']

for i in range(len(target_list)):
   TRAINING_CONFIG = {
       "bert_induced": True,
       "source_file": 'both',
       "target": target_list[i],
       "item": 'all',
       "batch_size": 512,
       'lr': 1e-4,
       "bert_model": 'bio_clinical_bert',
       "device_number": str(device),
       "bert_freeze": True,
       "only_BCE": True,
       "wandb_project_name": 'test-on-both',
   }

   TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

   # Run script
   subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)
