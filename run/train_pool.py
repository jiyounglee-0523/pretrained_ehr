import subprocess
import os

# Configuration before run

PATH = '/home/jylee/pretrained_ehr/rnn_model/'
SRC_PATH = PATH+'testonpooled.py'

device = 3
os.environ['CUDA_VISIBLE_DEVICES'] = str(device)


target_list = ['los>3day']

for i in range(len(target_list)):
   TRAINING_CONFIG = {
       "bert_induced": True,
       "source_file": 'both',
       "target_file": 'mimic',
       "target": target_list[i],
       "item": 'all',
       "batch_size": 512,
       'lr': 1e-4,
       "bert_model": 'bio_clinical_bert',
       "device_number": str(device),
       "bert_freeze": True,
       "only_BCE": True,
   }

   TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

   # Run script
   subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)
