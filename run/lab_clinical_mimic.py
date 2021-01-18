import subprocess
import os

# Configuration before run

PATH = '/home/jylee/pretrained_ehr/rnn_model/'
SRC_PATH = PATH+'main.py'

device = 3
os.environ['CUDA_VISIBLE_DEVICES'] = str(device)


target_list = ['readmission', 'mortality', 'los>3day', 'los>7day', 'dx_depth1_unique']

for target in target_list:
   TRAINING_CONFIG = {
       "bert_induced": True,
       "source_file": 'mimic',
       "item": 'all',
       "bert_freeze": True,
       "max_length": 300,
       "target": target,
       "bert_model": 'bio_clinical_bert',
       "device_number": device,
       "concat": True,
   }

   TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

   # Run script
   subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)
