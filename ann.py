import torch
import torch.nn as nn
import numpy as np
import itertools

""" Fully connected artificial neural network """
    
    
class ANN(torch.nn.Module):
    def __init__(self, config):
        super(ANN, self).__init__()

        input_size = config['input_size']
        output_size = config['output_size']
        num_hidden_layers = config['num_hidden_layers']
        neurons_per_layer = config['neurons_per_layer']

        self.input_layer = nn.Linear(input_size, neurons_per_layer)
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
        self.output_layer = nn.Linear(neurons_per_layer, output_size)
        
        self.config = config  # Store the config as an attribute

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        x = self.output_layer(x)
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