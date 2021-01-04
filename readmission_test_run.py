import subprocess
import os

# Configuration before run

PATH = '/home/jylee/pretrained_ehr/rnn_model/'
SRC_PATH = PATH+'test.py'

few_shot_list = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

for few_shot in few_shot_list:
    TRAINING_CONFIG = {
        "bert_induced": True,
        "source_file": 'eicu',
        "test_file": 'mimic',
        "few_shot": few_shot,
        "target": 'readmission',
        "bert_freeze": True,
        "device_number": 0
    }


    TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

    # Run script
    subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)
    print('finished few_shot {}'.format(few_shot))
