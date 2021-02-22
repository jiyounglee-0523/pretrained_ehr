import subprocess
import os

# Configuration before run

PATH = '/data/private/fixed/rnn_model/'
SRC_PATH = PATH+'main.py'

device = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(device)


target_list = ['readmission']

for target in target_list:
   TRAINING_CONFIG = {
       "bert_induced": True,
       "source_file": 'eicu',
       "item": 'all',
       "batch_size": 512,
       "bert_freeze": True,
       "target": target,
       "lr":1e-4,
       "bert_model": 'blue_bert',
       "device_number": device,
       "cls_freeze": True,
       "input_path": '/data/private/input_data/',
       "path": '/data/private/KDD_output2/',
       "only_BCE": True,
       "wandb_project_name": 'cls_fixed_train',
   }

   TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

   # Run script
   subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)
