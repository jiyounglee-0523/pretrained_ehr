import subprocess
import os

# Configuration before run

PATH = '/home/jylee/pretrained_ehr/rnn_model/'
SRC_PATH = PATH+'main.py'

device = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(device)



bert_model_list = ['bert_small']


for bert in bert_model_list:
   TRAINING_CONFIG = {
       "bert_induced": True,
       "source_file": 'both',
       "target": 'mortality',
       "item": 'lab',
       "bert_freeze": True,
       "bert_model": bert,
       "device_number": device,
       "input_path": '/home/jylee/from_ghhur/input_data_min_freq_zero/',
       "only_BCE": True,
       "transformer": True,
       "wandb_project_name": 'min_freq_zero',
       "not_removed_minfreq": True
   }

   TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

   # Run script
   subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)
