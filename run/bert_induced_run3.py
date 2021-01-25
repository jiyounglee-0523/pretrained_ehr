import subprocess
import os

# nonconcat version

# Configuration before run
device = '6'
os.environ['CUDA_VISIBLE_DEVICES'] = device

PATH = '/home/jylee/pretrained_ehr/rnn_model/'
SRC_PATH = PATH+'main2.py'

source_file_list = ['mimic', 'mimic', 'mimic']
target_list = ['mortality', 'mortality', 'mortality']
seed_list = [2020, 2023, 2029]
item_list = ['med', 'med', 'med']

for i in range(len(source_file_list)):
    TRAINING_CONFIG = {
        "bert_induced": True,
        "source_file": source_file_list[i],
        "target": target_list[i],
        "item": item_list[i],
        "bert_model": 'bio_bert',
        "bert_freeze": True,
        "device_number": device,
        "transformer": True,
        "concat": True,
        "seed": seed_list[i]
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