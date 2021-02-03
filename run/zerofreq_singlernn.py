import subprocess
import os

# Configuration before run
device = 3

os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

PATH = '/home/jylee/pretrained_ehr/rnn_model/'
SRC_PATH = PATH+'main.py'

item_list = ['med']
target_list = ['mortality']

for item in item_list:
    for target in target_list:
        TRAINING_CONFIG = {
            "source_file": 'both',
            "target": target,
            "item": item,
            "bert_model": 'pubmed_bert',
            "bert_freeze": True,
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
