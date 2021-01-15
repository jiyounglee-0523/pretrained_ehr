import subprocess
import os

# Configuration before run

PATH = '/home/jylee/pretrained_ehr/rnn_model/'
SRC_PATH = PATH+'main.py'

device = 6
os.environ['CUDA_VISIBLE_DEVICES'] = str(device)


target_list = ['readmission', 'mortality','los>3day']


for target in target_list:
   TRAINING_CONFIG = {
       "bert_induced": True,
       "source_file": 'mimic',
       "item": 'med',
       "bert_freeze": True,
       "target": target,
       "bert_model": 'bio_clinical_bert',
       "device_number": device,
   }

   TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

   # Run script
   subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)
