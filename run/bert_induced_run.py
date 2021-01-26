import subprocess
import os

# Configuration before run
device = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = device

PATH = '/home/jylee/pretrained_ehr/rnn_model/'
SRC_PATH = PATH+'main2.py'

#source_file_list = ['eicu', 'mimic']
#target_list = ['los>3day', 'dx_depth1_unique']
seed_list = [2020, 2021, 2026, 2028, 2029]

#for source_file in source_file_list:
#    for target in target_list:
for seed in seed_list:
    TRAINING_CONFIG = {
        "bert_induced": True,
        "source_file": 'mimic',
        "target": 'dx_depth1_unique',
        "item": 'med',
        "bert_model": 'bio_bert',
        "bert_freeze": True,
        "device_number": device,
        "transformer": True,
        #"concat": True,
        "seed": seed

    }


    TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

    # Run script
    subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)


# for source_file in source_file_list:
#     for target in target_list:
#         TRAINING_CONFIG = {
#             "bert_induced": True,
#             "source_file": source_file,
#             "target": target,
#             "item": 'med',
#             "bert_model": 'bio_bert',
#             "bert_freeze": True,
#             "device_number": device,
#             "transformer": True,
#             "cls_freeze": True,
#             "concat": True
#         }
#
#
#         TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]
#
#         # Run script
#         subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)
#         print('Finished {} {}'.format(source_file, target))