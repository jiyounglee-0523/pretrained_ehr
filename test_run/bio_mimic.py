import subprocess
import os

# Configuration before run

PATH = '/home/jylee/pretrained_ehr/rnn_model/'
SRC_PATH = PATH+'test.py'

device = 3
os.environ['CUDA_VISIBLE_DEVICES'] = str(device)


target_list = ['readmission', 'mortality', 'los>3day', 'los>7day', 'dx_depth1_unique']

for target in target_list:
   TRAINING_CONFIG = {
       "bert_induced": True,
       "source_file": 'mimic',
       "test_file": 'mimic',
       "few_shot": 0.0,
       "item": 'lab',
       "bert_freeze": True,
       "target": target,
       "bert_model": 'bio_bert',
       "device_number": device,
       "concat": True,
       #"cls_freeze": True
   }

   TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

   # Run script
   subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)
