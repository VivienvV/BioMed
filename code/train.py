import pandas as pd
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import shap
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.model_selection import GridSearchCV
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

class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.device = args.device
        self.LSTM_num_layers = args.LSTM_num_layers
        self.LSTM_hidden_size = args.LSTM_hidden_size
        self.lstm = nn.LSTM(args.input_size, args.LSTM_hidden_size, args.LSTM_num_layers, batch_first=True)
        self.fc = nn.Linear(args.LSTM_hidden_size, args.num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.LSTM_num_layers, x.size(0), self.LSTM_hidden_size).to(self.device) 
        c0 = torch.zeros(self.LSTM_num_layers, x.size(0), self.LSTM_hidden_size).to(self.device) 
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
    return train_arr, labels

def feature_selection(args, X_train, X_test, y_train, y_test, train_call):

    # remove the features beneath a threshold (removes 5 features)
    sel_variance_threshold = VarianceThreshold(.85 * (1 - .85)) 
    X_train_remove_variance = sel_variance_threshold.fit_transform(X_train)

    # remove the same features from the testing data
    X_test_remove_variance = sel_variance_threshold.transform(X_test)

    # select the 900 best features 
    model = SelectKBest(k=args.num_features)
    
    X_train = model.fit_transform(X_train_remove_variance, y_train)
    X_test = model.transform(X_test_remove_variance)
    cols = model.get_support(indices=True)
    
    return X_train, X_test, train_call.iloc[cols]

def feature_importance(args, model, X_train, X_test, fdf):
  features = (fdf["ID_no"]).tolist()

  sm = torch.load("model/{}.pth".format(args.model)).to(args.device)
  X_full = np.vstack((X_train, X_test))

  e = shap.DeepExplainer(sm, torch.from_numpy(X_full).to(args.device))
  shap_values = e.shap_values(torch.from_numpy(X_full).to(args.device))

  shap.summary_plot(shap_values, features=torch.from_numpy(X_full).to(args.device), feature_names = features, show=False, class_names=['HER2+', 'HR+', 'TN'])
  plt.savefig("summary_plot.png")


  her2 = np.mean(np.abs(shap_values[0]), axis=0)
  hr = np.mean(np.abs(shap_values[1]), axis=0)
  tn = np.mean(np.abs(shap_values[2]), axis=0)

  m = (np.mean(np.abs(shap_values), axis=0))
  mm = np.mean(m, axis=0)
  new = np.c_[features, fdf['Chromosome'].astype('int32'), fdf['Start'].astype('int32'), fdf['End'].astype('int32'),  mm , her2, hr, tn, fdf["Gene_IDs"]]   
  df = pd.DataFrame(new, columns=['feature_no', 'chromosome', 'start', 'end', 'mean_fi', 'HER2+_fi', 'HR+_fi', 'TN_fi', 'gene_ids']) 
  df = df.sort_values(by=['mean_fi'], ascending=False)
  df = df.astype({'feature_no': 'int32'})
  df = df.reset_index(drop=True)
  df.to_csv('feature_importance.csv')

  return

def main():
  parser = argparse.ArgumentParser()

  # Model params
  parser.add_argument('--model', type=str, default='NN',
                      help='Which model to use for training: NN or LSTM')
  parser.add_argument('--num_features', type=int, default=900,
                      help='Length of an input sequence/ amount of features each sample contains')
  parser.add_argument('--input_size', type=int, default=1,
                      help='Size of an input sequence')
  parser.add_argument('--LSTM_hidden_size', type=int, default=128,
                      help='Number of units in each LSTM layer')
  parser.add_argument('--LSTM_num_layers', type=int, default=2,
                      help='Number of hidden layers in the LSTM')
  parser.add_argument('--NN_hidden', type=list, default=[128,128],
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

  args, unknown = parser.parse_known_args()

  torch.manual_seed(42)

  if args.model == 'NN':
    model = NN(args, args.NN_hidden).to(args.device)
    params = {
        'optimizer__lr' : [0.001, 0.0001, 0.0003],
        'optimizer__weight_decay' : [0.1, 0.01, 0.001],
        'max_epochs' : [50, 100],
        'module__NN_hidden': [[450], [128], [128, 128]],
    }
    # params = {
    #     'optimizer__lr' : [0.001],
    #     'optimizer__weight_decay' : [0.1, 0.01],
    #     'max_epochs' : [50, 10],
    #     'module__NN_hidden': [[450]],
    # }

  elif args.model == 'LSTM':
    model = LSTM(args).to(args.device)
    params = {
        'optimizer__lr' : [0.001, 0.0001, 0.0003],
        'optimizer__weight_decay' : [0.1, 0.01, 0.001, 0.0001],
        'max_epochs' : [100, 150],
        'LSTM_hidden_size': [128, 450], 
        'LSTM_num_layers' : [1, 2]
    }  

  train_call = pd.read_csv('full_data.csv', delimiter=',' )
  train_clin = pd.read_csv('Train_clinical.txt', delimiter='\t' )
  train_arr, labels = preprocess_data(train_call, train_clin)

  net = NeuralNetClassifier(
    module=model, 
    criterion=torch.nn.CrossEntropyLoss, 
    max_epochs=args.num_epochs, 
    module__args=args,
    optimizer=optim.Adam, 
    )

  iter_acc = []
  iter_params = {}
  iter_est = {}

  for i in range(1,11):

    # START CROSS VALIDATION ITERATION
    kf = StratifiedKFold(n_splits=args.num_folds, shuffle=True)
    f = 1
    best_params = {}
    best_scores = {}
    best_est = {}

    # Split data into K (default=5) folds, perform feature selection 
    for train_index, test_index in kf.split(train_arr, labels):
      X_train, X_test = train_arr[train_index], train_arr[test_index]
      y_train, y_test = labels[train_index], labels[test_index]
      X_train, X_test, _ = feature_selection(args, X_train, X_test, y_train, y_test, train_call)
      net.set_params(train_split=False, verbose=0)
      # Perform grid search on 3 repeats
      gs = GridSearchCV(net, params, refit=True, cv=3, scoring='accuracy', verbose=0)
      gs.fit(X_train.astype('float32'), y_train.astype('int64'))

      print("FOLD {}: Mean best accuracy: {:.3f}, best params: {}".format(f, gs.best_score_, gs.best_params_))
      
      best_scores[f] = gs.best_score_
      best_params[gs.best_score_] = gs.best_params_
      best_est[gs.best_score_] = gs.best_estimator_
      f += 1

    av_acc = sum(best_scores.values())/len(best_scores)
    best_modelparams = best_params[max(best_scores.values())]
    best_estimator = best_est[max(best_scores.values())]


    print('\nITERATION {}: average accuracy score: {}, best params: {}\n'.format(i, av_acc, best_modelparams))

    iter_acc.append(av_acc)
    iter_params[av_acc] = best_modelparams
    iter_est[av_acc] = best_estimator

  print('--------------------------------')
  print('--------------------------------\nFinished cross-validation')
  print('Mean accuracy over all iterations: ', np.mean(np.array(iter_acc)))
  print('Best params over all iterations: ', iter_params[max(iter_acc)])

  best_model = iter_est[max(iter_acc)]

  # Save model to pickle file
  # with open('model/{}.pkl'.format(args.model), 'wb') as f:
  #   pickle.dump(best_model, f)


  best_model.save_params(f_params='model/{}.pkl'.format(args.model))

if __name__ == '__main__':
    main()
