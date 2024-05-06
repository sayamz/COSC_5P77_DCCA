import torch
import torch.nn as nn

class DccaNet(nn.Module):
    def __init__(self, input_dim_1, input_dim_2, hidden_dims, output_dim):
        super(DccaNet, self).__init__()
        
        # Define the network architectures for the two views
        self.view_1_layers = nn.ModuleList()
        self.view_2_layers = nn.ModuleList()
        
        # Input layers
        self.view_1_layers.append(nn.Linear(input_dim_1, hidden_dims[0]))
        self.view_2_layers.append(nn.Linear(input_dim_2, hidden_dims[0]))
        
        # Hidden layers
        for i in range(1, len(hidden_dims)):
            self.view_1_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.view_2_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            
        # Output layers
        self.view_1_layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.view_2_layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.relu = nn.ReLU()
        
    def forward(self, x1, x2):
        # Pass the inputs through the respective view networks
        for layer in self.view_1_layers:
            x1 = self.relu(layer(x1))
        for layer in self.view_2_layers:
            x2 = self.relu(layer(x2))
        
        return x1, x2

    #%% Train the model

#%%