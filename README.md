# submeso_ML_workflow
This repo contains code used to train a Neural Network parmeterization for submesoscale vertical buoyancy fluxes (Bodner, Balwada, and Zann, in prep.)

All modules can be found under `submeso_ml`, which consists of the following:
* `data` contains the dataset file used as our dataloader.
* `systems` is a general regression system which initiates the iteration process during training.
* `models` currently contains a Fully Convolutional Neural Network, or `fcnn`.

The folder `scripts` contains `train_fcnn_res_1_4.py` which interacts with the above modules to train the fcnn. The configuration of the cnn is defined by `config` within the code. The script is executed with `run_train_fcnn.sh`
