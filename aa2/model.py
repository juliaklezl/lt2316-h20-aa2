import torch.nn as nn
#import torch

# in demo we start by batching and shuffling, do we need this here???  --> no, in training loop

# start with GRU layer, since it seems to be the middle one in terms of complexity..

class ClassifierModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers): # add arguments as you need them
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.GRU = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)  # not sure what to do with hidden size for multiple layers..
        self.lin3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)  # is softmax the right thing to get probabilities for labels?

        
    def forward(self, batch, device):
        hidden = self.lin1(batch)
        hidden2 = self.lin2(hidden)
        gru, h_0 = self.GRU(hidden2)
        last = self.lin3(gru)
        out = self.softmax(last).to(device)
        return out
        
