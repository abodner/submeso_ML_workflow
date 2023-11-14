
import torch
import torch.nn as nn
import numpy as np
#import pyqg_explorer.util.transforms as transforms
import os
import pickle
import pytorch_lightning as pl

## From Andrew/Pavel's code, function to create a CNN block
def make_block(in_channels: int, out_channels: int, kernel_size: int, 
        ReLU = 'ReLU', batch_norm = True) -> list:
    '''
    Packs convolutional layer and optionally ReLU/BatchNorm2d
    layers in a list
    '''
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
        padding='same', padding_mode='reflect')
    block = [conv]
    if ReLU == 'ReLU':
        block.append(nn.ReLU())
    elif ReLU == 'SiLU':
        block.append(nn.SiLU())
    elif ReLU == 'LeakyReLU':
        block.append(nn.LeakyReLU(0.2))
    elif ReLU == 'False':
        pass
    else:
        print('Error: wrong ReLU parameter:')
    if batch_norm:
        block.append(nn.BatchNorm2d(out_channels))
    return block


class FCNN(nn.Module):
    def __init__(self,config):
        '''
        Packs sequence of n_conv=config["conv_layers"] convolutional layers in a list.
        First layer has config["input_channels"] input channels, and last layer has
        config["output_channels"] output channels
        '''
        super().__init__()
        self.config=config

        blocks = []
        ## If the conv_layers key is missing, we are running
        ## with an 8 layer CNN
        if ("conv_layer" in self.config) == False:
            self.config["conv_layers"]=8
        blocks.extend(make_block(self.config["input_channels"],128,self.config["kernel"],self.config["activation"])) #1
        blocks.extend(make_block(128,64,self.config["kernel_hidden"],self.config["activation"]))                            #2
        if self.config["conv_layers"]==3:
            blocks.extend(make_block(64,self.config["output_channels"],self.config["kernel_hidden"],'False',False))
        elif self.config["conv_layers"]==4:
            blocks.extend(make_block(64,32,self.config["kernel_hidden"],self.config["activation"]))                            
            blocks.extend(make_block(32,self.config["output_channels"],self.config["kernel_hidden"],'False',False))
        else: ## 5 layers or more
            blocks.extend(make_block(64,32,self.config["kernel_hidden"],self.config["activation"])) ## 3rd layer
            for aa in range(4,config["conv_layers"]):
                ## 4th and above layer
                blocks.extend(make_block(32,32,self.config["kernel_hidden"],self.config["activation"]))
            ## Output layer
            blocks.extend(make_block(32,self.config["output_channels"],self.config["kernel_hidden"],'False',False))
        self.conv = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv(x)
        return x

    def save_model(self):
        """ Save the model config, and optimised weights and biases. We create a dictionary
        to hold these two sub-dictionaries, and save it as a pickle file """
        if self.config["save_path"] is None:
            print("No save path provided, not saving")
            return
        save_dict={}
        save_dict["state_dict"]=self.state_dict() ## Dict containing optimised weights and biases
        save_dict["config"]=self.config           ## Dict containing config for the dataset and model
        save_string=os.path.join(self.config["save_path"],self.config["save_name"])
        with open(save_string, 'wb') as handle:
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Model saved as %s" % save_string)
        return





