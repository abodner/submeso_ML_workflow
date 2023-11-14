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



submeso_dataset=dataset.SubmesoDataset(['grad_B','FCOR', 'Nsquared', 'HML', 'TAU',
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
input_channels=10
output_channels=1
conv_layers = 3
kernel = 5
kernel_hidden =3
activation="ReLU"
arch="fcnn"
epochs=100
save_path=BASE+"trained_models/"
save_name="fcnn_k3_l3_select_1_4.pt"
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

#figure_fields=wandb.Image(plot_helpers.plot_fields(pyqg_dataset,model))
#wandb.log({"Fields": figure_fields})

#r2,corr=metrics.get_offline_metrics(model,test_loader)
# figure_power=wandb.Image(figure_power)
# Calculate r2 and corr for test dataset
r2_list = []
corr_list = []
mse_list=[]
for x_data, y_data in test_loader:
    output = model(x_data.to(device))
    mse = nn.MSELoss(y_data.to(device),output)
    r2 = r2_score(output.detach().cpu().numpy(), y_data.detach().cpu().numpy())
    corr, _ = pearsonr(output.detach().cpu().numpy().flatten(), y_data.detach().cpu().numpy().flatten())
    r2_list.append(r2.detach().cpu())
    corr_list.append(corr.detach().cpu())
    mse_list.append(mse.detach().cpu())



# Average r2 and corr values over the test dataset
r2_avg = torch.tensor(r2_list.detach().cpu()).mean().item()
corr_avg = torch.tensor(corr_list.detach().cpu()).mean().item()
test_loss = torch.tensor(mse_list.detach().cpu()).mean().item()

# wandb.log({"Power spectrum": figure_power})
wandb.run.summary["r2"]=r2_avg
wandb.run.summary["corr"]=corr_avg
wandb.run.summary["test_loss"]=test_loss
wandb.finish()
    
    



project_name="submeso_ML"







