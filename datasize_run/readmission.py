import subprocess
import os

# Configuration before run

PATH = '/data/private/fixed/rnn_model/'
SRC_PATH = PATH+'training_datasize_dependent.py'

device = 3
data_portion_list = [0.1]

os.environ['CUDA_VISIBLE_DEVICES'] = str(device)


for data_portion in data_portion_list:
    TRAINING_CONFIG = {
        "bert_induced": True,
        "source_file": 'eicu',
        "few_shot": data_portion,
        "target": 'los>7day',
        "item": 'all',
        "max_length": 150,
        "bert_model": 'bio_clinical_bert',
        "bert_freeze": True,
        "input_path": '/data/private/input_data/',
        "path": '/data/private/KDD_output2/',
        "device_number": device,
        "only_BCE": True,
        "wandb_project_name": 'new_rnn_datasizedependent'
    }

    TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k, v) for (k, v) in
                            list(TRAINING_CONFIG.items())]

    # Run script
    subprocess.run(['python', SRC_PATH] + TRAINING_CONFIG_LIST)