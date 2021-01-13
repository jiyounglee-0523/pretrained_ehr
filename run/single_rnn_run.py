import subprocess
import os

# Configuration before run
device = 7

os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

PATH = '/home/jylee/pretrained_ehr/rnn_model/'
SRC_PATH = PATH+'main.py'

source_file_list = ['mimic']
target_list = ['readmission', 'mortality', 'los>3day', 'los>7day', 'dx_depth1_unique']

for source_file in source_file_list:
    for target in target_list:
        TRAINING_CONFIG = {
            "source_file": source_file,
            "target": target,
            "item": 'inf',
            "bert_freeze": True,
            "device_number": device
        }


        TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

        # Run script
        subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)
        print('Finished {} {}'.format(source_file, target))