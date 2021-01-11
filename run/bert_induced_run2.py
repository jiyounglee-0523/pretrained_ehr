import subprocess
import os

# Configuration before run
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

PATH = '/home/jylee/pretrained_ehr/rnn_model/'
SRC_PATH = PATH+'main.py'

source_file_list = ['eicu', 'mimic']
target_list = ['los>3day']

for source_file in source_file_list:
    for target in target_list:
        TRAINING_CONFIG = {
            "bert_induced": True,
            "source_file": source_file,
            "target": target,
            "bert_freeze": True,
            "device_number": 0
        }


        TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

        # Run script
        subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)
        print('Finished {} {}'.format(source_file, target))