import subprocess
import os

# Configuration before run

PATH = '/home/jylee/pretrained_ehr/rnn_model/'
SRC_PATH = PATH+'test.py'

device = 0
few_shot_list = [1.0]

target_list = ['los>3day', 'los>7day', 'dx_depth1_unique']

os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

for target in target_list:
    TRAINING_CONFIG = {
        "source_file": 'eicu',
        "test_file": 'mimic',
        "few_shot": 1.0,
        "target": target,
        "bert_freeze": True,
        "item": 'inf',
        "max_length": 150,
        "device_number": device,
        #"concat": True,
        "bert_model": 'bio_bert'
    }

    TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k, v) for (k, v) in
                            list(TRAINING_CONFIG.items())]

    # Run script
    subprocess.run(['python', SRC_PATH] + TRAINING_CONFIG_LIST)
    print('finished file {}'.format(target))


