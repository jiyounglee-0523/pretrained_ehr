import subprocess
import os

# Configuration before run

PATH = '/home/jylee/pretrained_ehr/rnn_model/'
SRC_PATH = PATH+'test.py'

device = 2
few_shot_list = [0.5]


os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

for few_shot in few_shot_list:
    TRAINING_CONFIG = {
        "bert_induced": True,
        "source_file": 'eicu',
        "test_file": 'mimic',
        "few_shot": few_shot,
        "item": 'lab',
        "target": 'mortality',
        "bert_freeze": True,
        "device_number": device,
        "bert_model": 'bio_bert',
        #"concat": True,
    }

    TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k, v) for (k, v) in
                            list(TRAINING_CONFIG.items())]

    # Run script
    subprocess.run(['python', SRC_PATH] + TRAINING_CONFIG_LIST)
    print('finished few_shot {}'.format(few_shot))


