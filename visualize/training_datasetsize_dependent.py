import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def visualize_trainingdatasize(result_file:pd.DataFrame, source_file, test_file, bert_model, item, concat:bool):
    task_list = ['readmission', 'mortality', 'los>3day', 'los>7day', 'dx_depth1_unique']

    for task in task_list:
        # narrow down the dataset
        few_shot_sample = result_file[result_file['bert_model'] == bert_model]
        few_shot_sample = few_shot_sample[few_shot_sample['target'] == task]
        few_shot_sample = few_shot_sample[few_shot_sample['item'] == item]
        few_shot_sample = few_shot_sample[few_shot_sample['concat'] == concat]
        few_shot_sample = few_shot_sample[few_shot_sample['test_file'] == test_file]
        test2test_sample = few_shot_sample[few_shot_sample['source_file'] == test_file]
        few_shot_sample = few_shot_sample[few_shot_sample['source_file'] == source_file]

        # test auprc for each few shot ratio
        zero_shot = np.array(few_shot_sample[few_shot_sample['few_shot'] == 0].test_auprc.values.tolist())
        one_shot = np.array(few_shot_sample[few_shot_sample['few_shot'] == 0.1].test_auprc.values.tolist())
        three_shot = np.array(few_shot_sample[few_shot_sample['few_shot'] == 0.3].test_auprc.values.tolist())
        five_shot = np.array(few_shot_sample[few_shot_sample['few_shot'] == 0.5].test_auprc.values.tolist())
        seven_shot = np.array(np.array(few_shot_sample[few_shot_sample['few_shot'] == 0.7].test_auprc.values.tolist()))
        nine_shot = np.array(few_shot_sample[few_shot_sample['few_shot'] == 0.9].test_auprc.values.tolist())
        full_shot = np.array(few_shot_sample[few_shot_sample['few_shot'] == 1].test_auprc.values.tolist())

        # test2test value (baseline)
        test2test = np.array(test2test_sample.test_auprc.values.tolist())

        # plot the result
        x_axis = np.array([0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        mean_value = np.array([np.mean(zero_shot), np.mean(one_shot), np.mean(three_shot), np.mean(five_shot),
                               np.mean(seven_shot), np.mean(nine_shot), np.mean(full_shot)])

        std_value = np.array([np.std(zero_shot), np.std(one_shot), np.std(three_shot), np.std(five_shot),
                              np.std(seven_shot), np.std(nine_shot), np.std(full_shot)])

        # mean
        ax.plot(x_axis, mean_value, color='darkblue', label='{}->{}'.format(source_file, test_file))
        # std
        ax.plot(x_axis, std_value + mean_value, color='lavender')
        ax.plot(x_axis, mean_value - std_value, color='lavender')
        plt.fill_between(x_axis, mean_value + std_value, mean_value - std_value, facecolor='lavender')

        # test2test
        ax.plot(x_axis, np.array([np.mean(test2test)] * 7), label='{}->{}'.format(test_file, test_file), color='red')

        ax.set_xticks(x_axis, minor=False)
        ax.grid(which='major', axis='x')
        ax.grid(which='major', axis='y')

        plt.title('{}_{}'.format(task, 'test AUPRC'))

        ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1))