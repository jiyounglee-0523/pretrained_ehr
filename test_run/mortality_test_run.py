import subprocess
import os

# Configuration before run

PATH = '/home/jylee/pretrained_ehr/rnn_model/'
SRC_PATH = PATH+'test.py'

device = 4
few_shot_list = [0.0]

model_list = ['bio_clinical_bert', 'bio_bert', 'pubmed_bert', 'blue_bert']

os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

for model in model_list:
    TRAINING_CONFIG = {
        "bert_induced": True,
        "source_file": 'eicu',
        "test_file": 'eicu',
        "few_shot": 0.0,
        "target": 'mortality',
        "bert_freeze": True,
        "item": 'lab',
        "device_number": device,
        "bert_model": model
    }

    TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k, v) for (k, v) in
                            list(TRAINING_CONFIG.items())]

    # Run script
    subprocess.run(['python', SRC_PATH] + TRAINING_CONFIG_LIST)
    print('finished file {}'.format(model))


