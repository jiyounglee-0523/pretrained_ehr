import subprocess
import os


# Configuration before run
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

PATH = '/home/ehr_pretrained/rnn_model/'
SRC_PATH = PATH+'main.py'

TRAINING_CONFIG = {
    "source_file":"mimic",
    "target":"readmission",
    "item":"lab",
    "time_window":'12",
    "model_type":"gru",
    "batch_size":512,
    "embedding_dim":128,
    "hidden_dim":128,
    "rnn_bidirection": False,
    "n_epochs":5,
    "lr":1e-3,
    "max_length":1,
}
TRAINING_CONFIG_LIST = ["--{}".format(k) if (isinstance(v, bool) and (v)) else "--{}={}".format(k,v) for (k,v) in list(TRAINING_CONFIG.items())]

# Run script
subprocess.run(['python',SRC_PATH]+TRAINING_CONFIG_LIST)