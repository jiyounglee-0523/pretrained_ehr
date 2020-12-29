##  Directory Settings

```
pretrained_ehrcode
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
```
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


## Updates
**v1.0** : Initial commit

**v1.1** : bert-induced

​	v1.2:  v1.1 debuged

**v2.0 ** 

- target = 'dx_depth1_unique'  DONE
- source_file = 'both'  DONE 
- bert_freeze  DONE
- early stopping DONE


## To-Do
- work on multiple items
- work on ordering multiple item datasets
