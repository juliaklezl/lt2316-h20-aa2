import torch.nn as nn
#import torch

class ClassifierModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers): # add arguments as you need them
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.GRU = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)  
        self.lin3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)  

        
    def forward(self, batch, device):
        hidden = self.lin1(batch)
        gru, h_0 = self.GRU(hidden)
        last = self.lin3(gru)
        out = self.softmax(last).to(device)
        return out
        
