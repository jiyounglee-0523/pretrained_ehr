import pandas as pd
import numpy as np


def rawdata_generator(result_file:pd.DataFrame, target, source_file):
    # extract necessary information
    result_file = result_file.loc[:, ['Name', 'State', 'bert_freeze', 'bert_induced', 'bert_model', 'cls_freeze', 'concat', 'item', 'seed',
          'source_file', 'target','eicu_test_auprc', 'mimic_test_auprc', 'test_auprc']]

    # make model name (singleRNN, CLSfixed, CLSfinetune)
    conditionlist = [(result_file['bert_induced'] == False),
                     (result_file['cls_freeze'] == True) & (result_file['bert_induced'] == True),
                     (result_file['cls_freeze'] == False) & (result_file['bert_induced'] == True)]
    choicelist = ['singleRNN', 'CLSfixed', 'CLSfinetune']
    result_file['model_name'] = np.select(conditionlist, choicelist)

    # make dataset
    result_sample = result_file[result_file['target'] == target]
    result_df = pd.pivot_table(result_sample,
                               columns = ['source_file', 'model_name'],
                               index = ['item', 'concat', 'bert_model', 'seed'],
                               values = ['mimic_test_auprc', 'eicu_test_auprc'],
                               aggfunc=['mean'])
    return result_df