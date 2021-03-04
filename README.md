# Unifying Heterogeneous Electronic Health Record Systems via Clinical Text-Based Code Embedding (KDD 2021 Under Review)

This repository provides Pytorch code to implement DescEmb, a code-agnostic EHR predictive model. 



## File Explanation

main.py : To train a model 

datasize_dependent.py : To train a model varying the size of training dataset (corresponds to Sec 4.5 in the paper)

few_shot.py : To transfer the model differing ratios of the target dataset (corresponds to Sec 4.6 in the paper)

divide_and_conquer.py : To test separately trained model and test on pooled (corresponds to Sec 4.7 - Divide & Conquer in the paper)

./preprocessing : code for preprocessing both MIMIC-III and eICU

./visualize_results : code fore visualizing results in the paper



## Execution
```ar
python main.py \
--DescEmb \                       # otherwise, CodeEmb
--source_file = 'eicu' \ 
--target='readmission' \ 
--item='all' \ 
--time_window = '12' \ 
--batch_size = 512 \ 
--embedding_dim = 128 \ 
--hidden_dim = 128 \ 
--n_epochs = 100 \ 
--input_path = './data_folder/'
--path = './output' \
```



## Data Directory

```
input
└─ all
	├─ eicu_12_all_150_2020.pkl
	├─ eicu_12_all_150_2021.pkl
	├─ ...
	└─ mimic_12_all_150_2029.pkl

```



## Model Saving Directory

```
output
└─ all
	├─ singleRNN
	│  ├─ mimic
	│  │   ├─ readmission
	│  │   ├─ mortality
    │  │   ├─ los_3days
    │  │   ├─ los_7days
    │  │   └─ dx_depth1_unique
    │  └─ eicu 
    │      ├─ readmission
    │      ├─ mortality
    │      ├─ los_3days
    │      ├─ los_7days
    │      └─ dx_depth1_unique
    │
    └─  cls_learnable 
      ├─ mimic
      │   ├─ readmission
      │   ├─ mortality
      │   ├─ los_3days
      │   ├─ los_7days
      │   └─ dx_depth1_unique
      ├─ eicu
      │   ├─ readmission
      │   ├─ mortality
      │   ├─ los_3days
      │   ├─ los_7days
      │   └─ dx_depth1_unique
      └─ both
          ├─ readmission
          ├─ mortality
          ├─ los_3days
          ├─ los_7days
          └─ dx_depth1_unique

```
