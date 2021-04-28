import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader

class NN(nn.Module):
    def __init__(self, args):
        super(NN, self).__init__()
        self.network = nn.ModuleList()

        if args.NN_hidden == [0]:
            self.network.append(nn.Linear(args.input_size, args.num_classes))
        else:
          sizes = [args.input_size] + args.NN_hidden
          for input, output in zip(sizes, sizes[1:]):
              self.network.append(nn.Linear(input, output))
              self.network.append(nn.ReLU())

        self.network.append(nn.Linear(sizes[-1], args.num_classes))
        self.network = nn.Sequential(*self.network)

    def forward(self, x):
        out = self.network(x)
        return out


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.LSTM_num_layers = args.LSTM_num_layers
        self.LSTM_hidden_size = args.LSTM_hidden_size
        self.lstm = nn.LSTM(args.input_size, args.LSTM_hidden_size, args.LSTM_num_layers, batch_first=True)
        self.fc = nn.Linear(args.LSTM_hidden_size, args.num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.LSTM_num_layers, x.size(0), self.LSTM_hidden_size).to(args.device) 
        c0 = torch.zeros(self.LSTM_num_layers, x.size(0), self.LSTM_hidden_size).to(args.device) 
        out, _ = self.lstm(x, (h0,c0))  
        out = out[:, -1, :]
        out = self.fc(out)
        return out