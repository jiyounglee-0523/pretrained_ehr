import subprocess
import os

# Configuration before run
device = 4

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
            "item": 'all',
            "max_length": 300,
            "bert_model": 'pubmed_bert',
            "bert_freeze": True,
            "device_number": device,
            "concat": True,
            "debug": True
        }


        TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

        # Run script
        subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)
        print('Finished {} {}'.format(source_file, target))
        #
        # if source_file != 'both':
        #     TRAINING_CONFIG = {
        #         "source_file": source_file,
        #         "test_file": source_file,
        #         "few_shot": 0.0,
        #         "target": target,
        #         "item": 'all',
        #         "max_length": 300,
        #         "bert_model": 'pubmed_bert',
        #         "bert_freeze": True,
        #         "device_number": device,
        #         #"concat": True
        #     }
        #
        #     TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k, v) for
        #                             (k, v) in list(TRAINING_CONFIG.items())]
        #     subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)
