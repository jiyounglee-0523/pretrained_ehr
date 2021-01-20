import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def visualize_trainingdatasize(result_file:pd.DataFrame, source_file, item, concat:bool, task_list:list):
    #task_list = ['readmission', 'mortality', 'los>3day', 'los>7day', 'dx_depth1_unique']
    #task_list = ['mortality', 'los>3day', 'los>7day', 'dx_depth1_unique']

    fig = plt.figure(figsize=(15, 8))
    i=0
    for task in task_list:
        i+=1
        # narrow down the dataset
        few_shot_sample = result_file[result_file['target'] == task]
        few_shot_sample = few_shot_sample[few_shot_sample['item'] == item]
        few_shot_sample = few_shot_sample[few_shot_sample['concat'] == concat]
        few_shot_sample = few_shot_sample[few_shot_sample['source_file'] == source_file]

        # test auprc for bert_induced
        bert_induced_sample = few_shot_sample[few_shot_sample['bert_induced'] == True]
        bert_one_shot = np.array(bert_induced_sample[bert_induced_sample['few_shot'] == 0.1].test_auprc.values.tolist())
        bert_three_shot = np.array(bert_induced_sample[bert_induced_sample['few_shot'] == 0.3].test_auprc.values.tolist())
        bert_five_shot = np.array(bert_induced_sample[bert_induced_sample['few_shot'] == 0.5].test_auprc.values.tolist())
        bert_seven_shot = np.array(bert_induced_sample[bert_induced_sample['few_shot'] == 0.7].test_auprc.values.tolist())
        bert_nine_shot = np.array(bert_induced_sample[bert_induced_sample['few_shot'] == 0.9].test_auprc.values.tolist())

        assert len(bert_one_shot) == 10, "check the number of experiments, it exceeds 10"
        assert len(bert_three_shot) == 10, "check the number of experiments, it exceeds 10"
        assert len(bert_five_shot) == 10, "check the number of experiments, it exceeds 10"
        assert len(bert_seven_shot) == 10, "check the number of experiments, it exceeds 10"
        assert len(bert_nine_shot) == 10, "check the number of experiments, it exceeds 10"

        # test auprc for singleRNN
        single_sample = few_shot_sample[few_shot_sample['bert_induced'] == False]
        single_one_shot = np.array(single_sample[single_sample['few_shot'] == 0.1].test_auprc.values.tolist())
        single_three_shot = np.array(single_sample[single_sample['few_shot'] == 0.3].test_auprc.values.tolist())
        single_five_shot = np.array(single_sample[single_sample['few_shot'] == 0.5].test_auprc.values.tolist())
        single_seven_shot = np.array(single_sample[single_sample['few_shot'] == 0.7].test_auprc.values.tolist())
        single_nine_shot = np.array(single_sample[single_sample['few_shot'] == 0.9].test_auprc.values.tolist())
        assert len(single_one_shot) == 10, "check the number of experiments, it exceeds 10"
        assert len(single_three_shot) == 10, "check the number of experiments, it exceeds 10"
        assert len(single_five_shot) == 10, "check the number of experiments, it exceeds 10"
        assert len(single_seven_shot) == 10, "check the number of experiments, it exceeds 10"
        assert len(single_nine_shot) == 10, "check the number of experiments, it exceeds 10"

        # plot the result
        x_axis = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        ax = fig.add_subplot(2, 2, i)

        bert_mean_value = np.array([np.mean(bert_one_shot), np.mean(bert_three_shot), np.mean(bert_five_shot), np.mean(bert_seven_shot), np.mean(bert_nine_shot)])
        bert_std_value = np.array([np.std(bert_one_shot), np.std(bert_three_shot), np.std(bert_five_shot), np.std(bert_seven_shot), np.std(bert_nine_shot)])

        single_mean_value = np.array([np.mean(single_one_shot), np.mean(single_three_shot), np.mean(single_five_shot), np.mean(single_seven_shot), np.mean(single_nine_shot)])
        single_std_value = np.array([np.std(single_one_shot), np.std(single_three_shot), np.std(single_five_shot), np.std(single_seven_shot), np.std(single_nine_shot)])

        # bert-induced
        # mean
        ax.plot(x_axis, bert_mean_value, color='crimson', label='bert_induced')
        # std
        # ax.plot(x_axis, bert_std_value + bert_mean_value, color='lavender')
        # ax.plot(x_axis, mean_value - std_value, color='lavender')
        plt.fill_between(x_axis, bert_mean_value + bert_std_value, bert_mean_value - bert_std_value, facecolor='pink', alpha=0.5)

        # singleRNN
        ax.plot(x_axis, single_mean_value, color='darkblue', label='singleRNN')
        plt.fill_between(x_axis, single_mean_value + single_std_value, single_mean_value - single_std_value, facecolor='lightsteelblue', alpha=0.5)


        ax.set_xticks(x_axis, minor=False)
        ax.grid(which='major', axis='x')
        ax.grid(which='major', axis='y')

        plt.title('{}_{}_{}_{}'.format(task, item, concat, 'test AUPRC'))

    plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1))