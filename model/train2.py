import pandas as pd
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split, KFold
import shap
import matplotlib.pyplot as plt


class NN(nn.Module):
    def __init__(self, args):
        super(NN, self).__init__()
        self.network = nn.ModuleList()

        if args.NN_hidden == [0]:
            self.network.append(nn.Linear(args.num_features, args.num_classes))
        else:
          sizes = [args.num_features] + args.NN_hidden
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
        
    def forward(self, args, x):
        h0 = torch.zeros(self.LSTM_num_layers, x.size(0), self.LSTM_hidden_size).to(args.device) 
        c0 = torch.zeros(self.LSTM_num_layers, x.size(0), self.LSTM_hidden_size).to(args.device) 
        out, _ = self.lstm(x, (h0,c0))  
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def preprocess_data(train_call, train_clin):
    full_data = train_call
    train_arr = train_call[train_call.columns[-100:]].to_numpy()
    train_arr = train_arr.T
    l = train_clin[train_clin.columns[-1:]].to_numpy().T.squeeze()
    labels = np.array(pd.factorize(l)[0].tolist())
    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(train_arr, labels)
    clf.feature_importances_  
    model = SelectFromModel(clf, prefit=True)
    cols = model.get_support(indices=True)
    train_arr = model.transform(train_arr.astype(np.float32))
    
    return train_arr, labels, full_data.iloc[cols]

def train_and_test(args, model, train_loader, test_loader):

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  

  n_total_steps = len(train_loader)
  losslist = []
  acclist = []

  for epoch in range(args.num_epochs):
      n_correct = 0
      n_samples = 0
      tloss = 0
      # print(train_loader[0])
      for i, (arrayseq, labels) in enumerate(train_loader):  
          if args.model == 'LSTM':
            arrayseq = arrayseq.reshape(-1, args.num_features, 1)
          arrayseq = arrayseq.to(args.device)
          labels = labels.to(args.device)
          
          # Forward pass
          outputs = model(arrayseq)
          loss = criterion(outputs, labels)
          tloss += loss

          # Backward and optimize
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          _, predicted = torch.max(outputs.data, 1)
          n_samples += labels.size(0)
          n_correct += (predicted == labels).sum().item()
          
          if (i+1) % n_total_steps == 0:

            print (f'\rEpoch [{epoch+1}/{args.num_epochs}], Loss: {loss.item():.4f}, Accuracy: {100.0 * n_correct / n_samples}', end="")
            
      accuracy = 100.0 * n_correct / n_samples
      acclist.append(accuracy)
      losslist.append(tloss / n_total_steps)
  print("\nCompleted training!\n")
  torch.save(model, "model/{}.pth".format(args.model))

  with torch.no_grad():
      n_correct = 0
      n_samples = 0
      for arrayseq, labels in test_loader:
          if args.model == 'LSTM':
            arrayseq = arrayseq.reshape(-1, args.num_features, args.input_size)
          arrayseq = arrayseq.to(args.device)
          labels = labels.to(args.device)
          outputs = model(arrayseq)
          _, predicted = torch.max(outputs.data, 1)
          n_samples += labels.size(0)
          n_correct += (predicted == labels).sum().item()
      print(n_correct, 'correct of ', n_samples)
      testacc = 100.0 * n_correct / n_samples
      print(f'Accuracy of the network on the test Array sequences: {testacc} %')

  return acclist, losslist, testacc

def cross_validation(args, model, train_arr, labels):

  dataset =  TensorDataset(torch.from_numpy(train_arr), torch.from_numpy(labels))

  k_folds = 5
  # For fold results
  results = {}  
  # Define the K-fold Cross Validator
  kfold = KFold(n_splits=k_folds, shuffle=True)
    
  # Start print
  print('--------------------------------')

  # K-fold Cross Validation model evaluation
  for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    
    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')
    
    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=args.batch_size, sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=args.batch_size, sampler=test_subsampler)
    
    
    # Initialize optimizer
    
    _,_, acc = train_and_test(args, model, trainloader, testloader)
    results[fold] = acc
    
  # Print fold results
  print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
  print('--------------------------------')
  sum = 0.0
  for key, value in results.items():
    print(f'Fold {key}: {value} %')
    sum += value
  print(f'Average: {sum/len(results.items())} %')

  return


def main():
  parser = argparse.ArgumentParser()

  # Model params
  parser.add_argument('--model', type=str, default='LSTM',
                      help='Which model to use for training: NN or LSTM')
  parser.add_argument('--num_features', type=int, default=2834,
                      help='Length of an input sequence/ amount of features each sample contains')
  parser.add_argument('--input_size', type=int, default=1,
                      help='Size of an input sequence')
  parser.add_argument('--LSTM_hidden_size', type=int, default=128,
                      help='Number of units in each LSTM layer')
  parser.add_argument('--LSTM_num_layers', type=int, default=2,
                      help='Number of hidden layers in the LSTM')
  parser.add_argument('--NN_hidden', type=list, default=[128,128,128],
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
  parser.add_argument('--test_size', type=float, default=0.3,
                      help='Amount of data to use for testing (leave zero to use all data for training')
  parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cpu or gpu)')

  args, unknown = parser.parse_known_args()
  torch.manual_seed(42)

  train_call = pd.read_csv('full_data.csv', delimiter=',' )
  train_clin = pd.read_csv('Train_clinical.txt', delimiter='\t' )
  train_arr, labels, new_df = preprocess_data(train_call, train_clin)

  args.num_features = train_arr.shape[1]
  print("Number of features that will be used: ", args.num_features)
  if args.model == 'NN':
    model = NN(args).to(args.device)
  elif args.model == 'LSTM':
    model = LSTM(args).to(args.device)



  X_train, X_test, y_train, y_test = train_test_split(train_arr, labels, test_size=args.test_size)
  train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
  test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

  train_loader = torch.utils.data.DataLoader(train_data, 
                                            batch_size=args.batch_size,
                                            shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_data, 
                                            batch_size=args.batch_size, 
                                            shuffle=False)

  train_and_test(args, model, train_loader, test_loader)
  # cross_validation(args, model, train_arr, labels)

  features = (new_df["ID_no"]).tolist()
  gfeatures = (new_df["Gene_IDs"]).tolist()

  sm = torch.load("model/{}.pth".format(model)).to(args.device)
  X_full = np.vstack((X_train, X_test))
  e = shap.DeepExplainer(sm, torch.from_numpy(X_train).to(args.device))
  shap_values = e.shap_values(torch.from_numpy(X_full).to(args.device))


  class1 = shap_values[0]
  class2 = shap_values[1]
  class3 = shap_values[2]


  shap.summary_plot(shap_values, features=torch.from_numpy(X_train).to(args.device), feature_names = features, show=False)
  plt.savefig("summary_plot.png")

  m = (np.mean(np.abs(shap_values), axis=0))
  mm = np.mean(m, axis=0)
  new = np.c_[features, mm ]   
  df = pd.DataFrame(new, columns=['feature_no', 'feature_importance'])
  df['gene_ids'] = np.array(gfeatures)    
  df = df.sort_values(by=['feature_importance'], ascending=False)
  df = df.astype({'feature_no': 'int32'})
  df = df.reset_index(drop=True)

  df.to_csv('feature_importance.csv')
  # df['gene_ids'] = np.array(gfeatures)    
  # print(df)


  # df.to_csv('final.csv')

  

  
if __name__ == '__main__':
    main()