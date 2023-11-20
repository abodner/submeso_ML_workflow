#!/usr/bin/env python
# coding: utf-8

# ## Lightning FCNN hyper-parameter sweep 

import wandb
import numpy as np
import sys
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import torch.nn as nn



wandb.login()



BASE = '/scratch/ab10313/pleiades/'
PATH_NN= BASE+'NN_data_smooth/'



import submeso_ml.systems.regression_system as regression_system
import submeso_ml.models.fcnn as fcnn
import submeso_ml.data.dataset as dataset

# Define X,Y pairs (state, subgrid fluxes) for local network.local_torch_dataset = Data.TensorDataset(
BATCH_SIZE = 64  # Number of sample in each batch



submeso_dataset=dataset.SubmesoDataset(['grad_B','FCOR', 'HML', 'TAU',
              'Q', 'HBL', 'div', 'vort', 'strain'],res='1_4')


train_loader=DataLoader(
    submeso_dataset,
    num_workers=10,
    batch_size=64,
    sampler=SubsetRandomSampler(submeso_dataset.train_ind))

test_loader=DataLoader(
    submeso_dataset,
    num_workers=10,
    batch_size=len(submeso_dataset.test_ind),
    sampler=submeso_dataset.test_ind)





# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')





seed=123
batch_size=256
input_channels=9
output_channels=1
conv_layers = 3
kernel = 5
kernel_hidden =3
activation="ReLU"
arch="fcnn"
epochs=100
save_path=BASE+"trained_models/"
save_name="fcnn_k5_l3_res_1_4.pt"
lr=0.00024594159283761457
wd=0.023133758465751404

## Wandb config file
config={"seed":seed,
        "lr":lr,
        "wd":wd,
        "batch_size":batch_size,
        "input_channels":input_channels,
        "output_channels":output_channels,
        "activation":activation,
        "save_name":save_name,
        "save_path":save_path,
        "arch":arch,
        "conv_layers":conv_layers,
        "kernel":kernel,
        "kernel_hidden":kernel_hidden,
        "epochs":epochs}



wandb.init(project="submeso_ML",config=config)
model=fcnn.FCNN(config)
config["learnable parameters"]=sum(p.numel() for p in model.parameters())
system=regression_system.RegressionSystem(model,wandb.config["lr"],wandb.config["wd"])
wandb.watch(model, log_freq=1)
wandb_logger = WandbLogger()

trainer = pl.Trainer(
    default_root_dir=model.config["save_path"],
    accelerator="auto",
    max_epochs=config["epochs"],
    enable_progress_bar=False,
    logger=wandb_logger,
    )
trainer.fit(system, train_loader, test_loader)
#model.save_model()
torch.save(model, config["save_path"] + config["save_name"])

wandb.finish()
    

project_name="submeso_ML"







