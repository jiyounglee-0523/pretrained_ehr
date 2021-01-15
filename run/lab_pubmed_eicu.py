import subprocess
import os

# Configuration before run

PATH = '/home/jylee/pretrained_ehr/rnn_model/'
SRC_PATH = PATH+'main.py'

device = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

file_list = ['mimic', 'eicu', 'eicu', 'eicu', 'eicu', 'mimic', 'mimic', 'eicu', 'mimic', 'mimic']

target_list = ['mortality', 'mortality', 'readmission', 'readmission', 'readmission', 'readmission', 'readmission',
               'readmission', 'readmission', 'readmission']

bert_model_list = ['pubmed_bert', 'bio_clinical_bert', 'pubmed_bert', 'blue_bert', 'bio_bert', 'bio_clinical_bert', 'bio_bert',
                   'bio_clinical_bert', 'pubmed_bert', 'blue_bert']
seed_list = [2020, 2020, 2022, 2022, 2022, 2022, 2022, 2021, 2021, 2020]

for i in range(len(file_list)):
   TRAINING_CONFIG = {
       "bert_induced": True,
       "source_file": file_list[i],
       "item": 'med',
       "bert_freeze": True,
       "target": target_list[i],
       "bert_model": bert_model_list[i],
       "device_number": device,
       "seed": seed_list[i],
   }

   TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

   # Run script
   subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)
