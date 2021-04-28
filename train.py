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
      for i, (arrayseq, labels) in enumerate(train_loader):  
          if args.model == 'LSTM':
            arrayseq = arrayseq.reshape(-1, args.num_features, 1)
          arrayseq = arrayseq.to(device)
          labels = labels.to(device)
          
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
  torch.save(model)

  with torch.no_grad():
      n_correct = 0
      n_samples = 0
      for arrayseq, labels in test_loader:
          if args.model == 'LSTM':
            arrayseq = arrayseq.reshape(-1, args.num_features, args.input_size)
          arrayseq = arrayseq.to(device)
          labels = labels.to(device)
          outputs = model(arrayseq)
          _, predicted = torch.max(outputs.data, 1)
          n_samples += labels.size(0)
          n_correct += (predicted == labels).sum().item()
      print(n_correct, 'correct of ', n_samples)
      testacc = 100.0 * n_correct / n_samples
      print(f'Accuracy of the network on the test Array sequences: {testacc} %')

  return acclist, losslist, testacc

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
    parser.add_argument('--num_epochs', type=float, default=120,
                        help='Amount of epochs used in training')

    args, unknown = parser.parse_known_args()



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Load data into PyTorch
X_train, X_test, y_train, y_test = train_test_split(train_arr, labels, test_size=0.3)
train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

train_loader = torch.utils.data.DataLoader(train_data, 
                                           batch_size=args.batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, 
                                          batch_size=args.batch_size, 
                                          shuffle=False)




if __name__ == '__main__':
    main()