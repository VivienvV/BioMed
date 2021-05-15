#!/usr/bin/env python3
"""Reproduce your result by your saved model.

This is a script that helps reproduce your prediction results using your saved
model. This script is unfinished and you need to fill in to make this script
work. If you are using R, please use the R script template instead.

The script needs to work by typing the following commandline (file names can be
different):

python3 run_model.py -i unlabelled_sample.txt -m model.pkl -o output.txt

"""

# author: Chao (Cico) Zhang
# date: 31 Mar 2017

import argparse
import sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold
import numpy as np
from skorch import NeuralNetClassifier

class NN(nn.Module):
    def __init__(self, args, NN_hidden):
        super(NN, self).__init__()
        self.network = nn.ModuleList()

        if args.NN_hidden == [0]:
            self.network.append(nn.Linear(args.num_features, args.num_classes))
        else:
          sizes = [args.num_features] + NN_hidden
          for input, output in zip(sizes, sizes[1:]):
              self.network.append(nn.Linear(input, output))
              self.network.append(nn.ReLU())

        self.network.append(nn.Linear(sizes[-1], args.num_classes))
        self.network = nn.Sequential(*self.network)

    def forward(self, x):
        out = self.network(x)
        return out


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Reproduce the prediction')
    parser.add_argument('-i', '--input', required=True, dest='input_file',
                        metavar='unlabelled_sample.txt', type=str,
                        help='Path of the input file')
    parser.add_argument('-m', '--model', required=True, dest='model_file',
                        metavar='model.pkl', type=str,
                        help='Path of the model file')
    parser.add_argument('-o', '--output', required=True,
                        dest='output_file', metavar='output.txt', type=str,
                        help='Path of the output file')
    parser.add_argument('--num_features', type=int, default=900,
                        help='Length of an input sequence/ amount of features each sample contains')
    parser.add_argument('--input_size', type=int, default=1,
                        help='Size of an input sequence')
    parser.add_argument('--LSTM_hidden_size', type=int, default=128,
                        help='Number of units in each LSTM layer')
    parser.add_argument('--LSTM_num_layers', type=int, default=2,
                        help='Number of hidden layers in the LSTM')
    parser.add_argument('--NN_hidden', type=list, default=[450],
                        help='List of which the length is the number of hidden layers and the values are the layer sizes in the NN')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of classes the model needs to be able to predict')

    # Training params
    parser.add_argument('--batch_size', type=int, default=7,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Amount of epochs used in training')
    parser.add_argument('--num_folds', type=int, default=3,
                        help='Amount of folds for outer CV loop')  
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cpu or gpu)')
    # Parse options
    args = parser.parse_args()

    if args.input_file is None:
        sys.exit('Input is missing!')

    if args.model_file is None:
        sys.exit('Model file is missing!')

    if args.output_file is None:
        sys.exit('Output is not designated!')

    # Start your coding
    # suggested steps
    # Step 1: load the model from the model file
    net = NeuralNetClassifier(
        module=NN(args, args.NN_hidden).to(args.device), 
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=optim.Adam, 
        )

    net.initialize()  # This is important!
    net.load_params(f_params=args.model_file)

    fi = pd.read_csv('feature_importance.csv', delimiter=',')
    fi = fi.iloc[:-7]
    fi = fi['feature_no'].tolist()

    # Step 2: apply the model to the input file to do the prediction
    val_call = pd.read_csv(args.input_file, delimiter='\t' )
    val_call = val_call.iloc[fi]
    val_arr = val_call[val_call.columns[-57:]].to_numpy()
    val_arr = val_arr.T

    predictions = net.predict(val_arr.astype('float32'))   

    # Step 3: write the prediction into the desinated output file
    result = ['HER2+' if x == 0 else 'HR+' if x == 1 else 'TN' for x in predictions]
    df = pd.DataFrame(np.c_[list(val_call.columns[-57:].values), result], columns = ['Sample', 'Subgroup'])
    df.to_csv(args.output_file, sep='\t')

    # End your coding




if __name__ == '__main__':
    main()
