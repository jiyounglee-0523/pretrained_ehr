import pandas as pd
import numpy as np

def test_auprc(result_file:pd.DataFrame, bert_induced:bool, source_file, bert_model, item, concat:bool):
    task_list = ['readmission', 'mortality', 'los>3day', 'los>7day', 'dx_depth1_unique']
    if bert_induced:
        print('source file: {}, bert_model: {}, item: {}_{}'.format(source_file, bert_model, item, concat))
        print('')
    elif not bert_induced:
        print('source file: {}, singleRNN, item: {}_{}'.format(source_file, bert_model, item, concat))
        print('')

    for task in task_list:
        result_sample = result_file[result_file['bert_model'] == bert_model]
        result_sample = result_sample[result_sample['target'] == task]
        result_sample = result_sample[result_sample['item'] == item]
        result_sample = result_sample[result_sample['concat'] == concat]
        result_sample = result_sample[result_sample['test_file'] == source_file]
        result_sample = result_sample[result_sample['source_file'] == source_file]
        result_sample = result_sample[result_sample['bert_induced'] == bert_induced]

        test_auprc = np.array(result_sample.test_auprc.values.tolist())
        assert len(test_auprc) == 10, "check the number of experiments, it exceeds 10."

        test_mean = np.mean(test_auprc)
        test_std = np.std(test_auprc)

        print('{} mean: {:.3f}'.format(task, test_mean))
        print('{} std: {:.3f}'.format(task, test_std))
        print('')