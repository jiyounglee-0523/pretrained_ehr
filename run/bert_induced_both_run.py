import subprocess
import os

# Configuration before run

PATH = '/home/jylee/pretrained_ehr/rnn_model/'
SRC_PATH = PATH+'main.py'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


target_list = ['readmission', 'mortality', 'los>3day']
embedding_list = [512, 512, 768]
hidden_list = [256, 768, 768]

for i in range(len(target_list)):
   TRAINING_CONFIG = {
       "bert_induced": True,
       "source_file": 'both',
       "target": target_list[i],
       "bert_freeze": True,
       "device_number": 0,
       "embedding_dim": embedding_list[i],
       "hidden_dim": hidden_list[i]
   }

   TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

   # Run script
   subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)
   print('Finished {}'.format(target_list[i]))