import subprocess
import os

# Configuration before run

PATH = '/home/jylee/pretrained_ehr/rnn_model/'
SRC_PATH = PATH+'main.py'

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'


target_list = ['los>3day', 'mortality', 'los>3day']
source_file_list = ['mimic', 'eicu', 'eicu']

for i in range(len(target_list)):
   TRAINING_CONFIG = {
       "bert_induced": True,
       "source_file": source_file_list[i],
       "target": target_list[i],
       "batch_size": 48,
       "bert_model": 'bio_clinical_bert',
       "device_number": '0,1,2,3,4,5,6,7',
       "concat": True,
   }

   TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

   # Run script
   subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)
   print('Finished {}'.format(target_list[i]))