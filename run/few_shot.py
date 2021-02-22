import subprocess
import os

# Configuration before run

PATH = '/data/private/fixed/rnn_model/'
SRC_PATH = PATH+'test.py'

device = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(device)


few_shot_list = [1.0]

for few_shot in few_shot_list:
   TRAINING_CONFIG = {
       "bert_induced": True,
       "source_file": 'eicu',
       "test_file": 'mimic',
       "few_shot": few_shot,
       "item": 'all',
       "batch_size": 512,
       "bert_freeze": True,
       "target": 'dx_depth1_unique',
       "lr":1e-4,
       "bert_model": 'bio_clinical_bert',
       "device_number": device,
       "cls_freeze": True,
       "input_path": '/data/private/input_data/',
       "path": '/data/private/KDD_output2/',
       "only_BCE": True,
       "wandb_project_name": 'cls_fixed_few_shot',
   }

   TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

   # Run script
   subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)
