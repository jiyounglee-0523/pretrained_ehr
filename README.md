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
python main.py --args
or
python run.py   (working on run.py, not recommended)
```


## Updates
**v1.0** : Initial commit

**v1.1** : bert-induced

​     **v1.2** : v1.1 debuged


## To-Do
- work on *source_file = Total*
- work on multiple items
- work on ordering multiple item datasets
