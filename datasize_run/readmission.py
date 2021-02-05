import subprocess
import os

# Configuration before run

PATH = '/home/jylee/pretrained_ehr/rnn_model/'
SRC_PATH = PATH+'training_datasize_dependent.py'

device = 0
data_portion_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

os.environ['CUDA_VISIBLE_DEVICES'] = str(device)


for data_portion in data_portion_list:
    TRAINING_CONFIG = {
        "bert_induced": True,
        "source_file": 'eicu',
        "few_shot": data_portion,
        "target": 'readmission',
        "item": 'all',
        "max_length": 150,
        "bert_model": 'bio_clinical_bert',
        "bert_freeze": True,
        "path": '/home/jylee/data/pretrained_ehr/output/KDD_output/',
        "device_number": device,
        "only_BCE": True,
        "input_path": './',
        "wandb_project_name": 'new_rnn_datasizedependent'
    }

    TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k, v) for (k, v) in
                            list(TRAINING_CONFIG.items())]

    # Run script
    subprocess.run(['python', SRC_PATH] + TRAINING_CONFIG_LIST)