import subprocess
import os

# Configuration before run

PATH = '/home/jylee/pretrained_ehr/rnn_model/'
SRC_PATH = PATH+'training_datasize_dependent.py'

device = 7
data_portion_list = [0.1, 0.3, 0.5, 0.7, 0.9]

os.environ['CUDA_VISIBLE_DEVICES'] = str(device)


for data_portion in data_portion_list:
    TRAINING_CONFIG = {
        "bert_induced": True,
        "source_file": 'eicu',
        "few_shot": data_portion,
        "target": 'mortality',
        "item": 'med',
        "max_length": 150,
        "bert_model": 'bio_bert',
        "bert_freeze": True,
        "path": '/home/jylee/data/pretrained_ehr/output/KDD_output/',
        #"concat": True,
        "device_number": device,
    }

    TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k, v) for (k, v) in
                            list(TRAINING_CONFIG.items())]

    # Run script
    subprocess.run(['python', SRC_PATH] + TRAINING_CONFIG_LIST)
    print('finished few_shot {}'.format(data_portion))