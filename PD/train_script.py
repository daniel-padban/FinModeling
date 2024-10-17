from git import Tree
import torch
import torch.utils.data 
from torch.utils.data import DataLoader
import wandb
from model_def import SkipNN
from dataset_def import HMEQ_data
from trainer_def import ModelTrainer
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import datetime
import warnings
warnings.filterwarnings("ignore", message="WARNING:root:No positive instances have been seen in target. Recall is converted from NaN to 0s.")

device = torch.device(
    'cuda' 
    if torch.cuda.is_available()
    else 'mps' 
    if torch.backends.mps.is_available() 
    else 'cpu')

r_seed = 100
torch.manual_seed(r_seed)
torch.cuda.manual_seed_all(r_seed)
torch.mps.manual_seed(r_seed)

def read_json(path):
    with open(path,'r') as json_file:
        json_dict = json.load(json_file)
    return json_dict

config_dict = read_json('PD/config.json')

run_name = 'r1'
run = wandb.init(name=run_name,project='IRB_ML',config=config_dict)

train_p = float(run.config['train_p'])
test_p = 1-train_p
full_dataset = pd.read_csv('PD/pd_data/hmeq.csv')
train_set, test_set = train_test_split(full_dataset,test_size=test_p,random_state=r_seed)

train_dataset = HMEQ_data(data_df=train_set)
test_dataset = HMEQ_data(data_df=test_set)

batch_size = run.config['batch_size']
train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True, drop_last=True)

h1 = int(run.config['h1'])
h2 = int(run.config['h2'])
h3 = int(run.config['h3'])
s1 = int(run.config['s1'])

model = SkipNN(
    in_feats=train_dataset.num_feats,
    h1=h1,
    h2=h2,
    h3=h3,
    s1=s1,
    out=1,
    )
model.to(device=device)
run.watch(model,log='all',log_freq=100)

trainer = ModelTrainer(run=run, model=model,train_dataloader=train_dataloader,test_dataloader=test_dataloader,device=device,report_freq=100)

trainer.full_epoch_loop()
model_dest = f'PD/models/{datetime.datetime.now()}'
torch.save(trainer.model.state_dict(),model_dest)

run.finish(0)