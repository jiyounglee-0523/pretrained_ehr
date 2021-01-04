import subprocess
import os

# Configuration before run

PATH = '/home/jylee/pretrained_ehr/rnn_model/'
SRC_PATH = PATH+'main2.py'

source_file_list = ['mimic', 'eicu', 'both']
target_list = ['readmission']

for source_file in source_file_list:
    for target in target_list:
        TRAINING_CONFIG = {
            "bert_induced": True,
            "source_file": source_file,
            "target": target,
            "bert_freeze": True,
            "device_number": 4
        }


        TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

        # Run script
        subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)
        print('Finished {} {}'.format(source_file, target))