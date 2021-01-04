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




or
python run.py   (currently working on run.py, not recommended)
```



## Model Saved Directory

```
arxiv_output
├─ bert_induced_False (means single_rnn)
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
├─ bert_freeze 
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




## To-Do
- work on multiple items
- work on ordering multiple item datasets
- [Dataset] check if lengths of padded *item_id* and *item_offset_order* are the same.

- change wandb init (reinint=True)    for seed 