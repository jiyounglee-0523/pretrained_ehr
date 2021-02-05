import subprocess
import os

# Configuration before run

PATH = '/home/jylee/pretrained_ehr/rnn_model/'
SRC_PATH = PATH+'main.py'

device = 5
os.environ['CUDA_VISIBLE_DEVICES'] = str(device)


target_list = ['los>3day']
source_file_list = ['both']

for i in range(len(target_list)):
   TRAINING_CONFIG = {
       "source_file": source_file_list[i],
       "target": target_list[i],
       "item": 'all',
       "batch_size": 256,
       'lr': 1e-4,
       "bert_model": 'bio_clinical_bert',
       "device_number": str(device),
       "bert_freeze": True,
       "only_BCE": True,
       "wandb_project_name": 'all_RNN_lr',
   }

   TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

   # Run script
   subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)
