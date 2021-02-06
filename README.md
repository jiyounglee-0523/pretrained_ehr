##  Directory Settings

```
pretrained_ehr_code
├─ dataset
│  ├─ singlernn_dataloader.py
│  └─ prebert_dataloader 
├─ models
│  ├─ rnn_models.py
│  └─ prebert.py
├─ trainer
│  ├─ base_trainer.py
│  └─ bert_induced_trainer.py
├─ utils
│  ├─ embedding.py
│  └─ loss.py
├─ main.py
└─ run.py
```

## Execution
```ar
python main.py \
-- bert_induced = True \ 
--source_file = 'eicu' \ 
--target = 'readmission' \ 
--item = 'lab' \ 
--time_window = '12' \ 
--rnn_model_type = 'gru' \ 
--batch_size = ??? \ 
--embedding_dim = 128 \ 
--hidden_dim = 128 \ 
--rnn_bidirection = True \
--n_epochs = 50 \ 
--path = '/home/jylee/data/pretrained_ehr/output/' \

* maxmimum batch size is 128, otherwise it will show 'Out of Memory' error.

OR

(RECOMMEND BELOW)

for singleRNN
python run/single_rnn_run.py

for cls_learnable
python run/bert_induced_run.py

```



## Model Saved Directory

```
KDD_output
└─ lab / med / inf
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
    ├─ cls_learnable 
    │  ├─ mimic
    │  │   ├─ readmission
    │  │   ├─ mortality
    │  │   ├─ los_3days
    │  │   ├─ los_7days
    │  │   └─ dx_depth1_unique
    │  ├─ eicu
    │  │   ├─ readmission
    │  │   ├─ mortality
    │  │   ├─ los_3days
    │  │   ├─ los_7days
    │  │   └─ dx_depth1_unique
    │  └─ both
    │      ├─ readmission
    │      ├─ mortality
    │      ├─ los_3days
    │      ├─ los_7days
    │      └─ dx_depth1_unique
    │
    └─ bert_finetune
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



## Updates

**v1.0** : Initial commit

**v1.1** : bert-induced

​	v1.2:  v1.1 debuged

**v2.0** :

- target = 'dx_depth1_unique'  DONE
- source_file = 'both'  DONE 
- bert_freeze  DONE
- early stopping DONE

v2.1 : path saved, argparser debug

v2.2 : bert embedding from dictionary (takes too long for training)



**v3.0**: test.py (few_shot, zero_shot, interchange)

v3.1: changed code to new seed number, run files, train 

v3.2 : modified test.py / fixed *_run.py

v3.3 : save model at best_auprc (not best_eval_loss)



**v4.0**: cls_learnable, run.py fixed

v4.1: Added preprocessing files (Preprocessing_{1, 1_5, 2}.ipynb) 

v4.2: Added 4 BERTs (clinical, bio, blue, pubmed), changed BERT fintuning code to enable multi-GPU process, output_dir path changed, 3 items (lab, med, inf can be implemented)

v4.3: Revised nn.DataParallel for multi-gpu using / flatten warning should be fixed in future.

v4.4: 

​	test.py: debuged,  implemented partial parameter load

​	main.py : implemented *lab_concat*, added parser (concat, debug)

**v5.0**

​	test.py: few_shot - partially load parameters (treat differently for overlapping codes)

​	main.py: both - treat the same for overlapping codes

v5.1 main.py: cls_fixed, all

v5.2

​	test.py : implemented testing cls_freeze

​	training_datasize_dependent.py : training datasetsize dependent code

v5.3  singlernn_dataloader.py : implemented both

v5.4: added option for only BCE loss (vs. BCE + Focal); modified main.py, trainer/*

v5.5: singleRNN dataloader, visualize result 

v5.6: debug BCELoss 

v5.7: training dataset size dependent debug, visualize result 

v5.8 : visualize result/*

v5.9: test.py : debuged when source_file = both

**v6.0** : test.py: debuged when source_file = both (loading model parameters)

v6.1: main.py : implemented test on test dataset in main.py

v6.2: main.py: changed *input_path* as argument

**v7.0**: implemented transformers, fixed test error in BCE 

v7.1: implemented BERT finetuning for RNN models

v7.2: added label processing files

v7.3 : Bert mini Bert Tiny

v7.4: Bert Tiny finetune

v.7.5: Transformer - Bert tiny, mini, base debuged

v7.6: few-shot transformer implemented

v7.7: no early-stopping

v7.8 early-stopping with 100, seperate both evaluation

v7.9 : few-shot, data dependent experiment with transformer, seperate both evaluation

v8.0: segment embedding, no min freq

v8.1: removed segment and min freq, implemented lr scheduler

v8.2: retain best model and save at last

v8.3: removed v8.2, fixed some datasize dependent  

v8.4: changed few_shot so we can run now

v8.5: implemented pretrain on both and finetune individually

v8.6: few-shot and datasize dependent for transformer

v8.7: debugged pool -> finetune

v8.8: pool_with_base.py   -> 100% mimic + 10% eicu