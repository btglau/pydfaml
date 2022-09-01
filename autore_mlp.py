'''

'simple' MLP for autore classification
aug 9 2022 bryan lau

'''

import sys
import numpy as np
import pandas as pd

from sklearn import model_selection
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from autore_scikitsvc import getArgs, getData

class AREDataset(Dataset):
    def __init__(self, X, y, ndfa, transform=None, target_transform=None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        if ndfa > 2:
            self.y = torch.nn.functional.one_hot(self.y)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        rxn = self.X[idx,:]
        dfa = self.y[idx]
        if self.transform:
            rxn = self.transform(rxn)
        if self.target_transform:
            dfa = self.target_transform(dfa)
        return rxn, dfa

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch == 1:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, ndfa):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            if ndfa > 2:
                correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()
            else:
                correct += ((torch.sigmoid(pred) >= .5) == y).type(torch.float).sum().item()
                #correct += ((pred >= .5) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self,insize,ndfa):
        super().__init__()
        self.flatten = nn.Flatten()

        if ndfa > 2:
            outsize = ndfa
        else:
            outsize = 1
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(insize, 128, bias=True),
            nn.ReLU(),
            #nn.Linear(128, 128, bias=True),
            #nn.ReLU(),
            nn.Linear(128, outsize, bias=True)
        )

    def forward(self, x):
        # don't need to flatten 1D input
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits.squeeze()

if __name__ == '__main__':

    args = getArgs(sys.argv[1:])
    ndfa = len(args.dfa)

    X, y, Xlabel, Ylabel, summary, ARE = getData(args)

    # create Datasets for torch
    X_train, X_test, y_train, y_test, ind_train, ind_test = model_selection.train_test_split(
                                        X, y, range(X.shape[0]), test_size=0.2, random_state=0, stratify=y)
    if args.a == 1:
        transformer = preprocessing.MaxAbsScaler(copy=False).fit(X_train)
        transformer.transform(X_train)
        transformer.transform(X_test)
    training_data = AREDataset(X_train,y_train,ndfa)
    test_data = AREDataset(X_test,y_test,ndfa)

    # Create data loaders.
    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for Xt, yt in test_dataloader:
        print(f"Shape of X: {Xt.shape} {Xt.dtype}")
        print(f"Shape of y: {yt.shape} {yt.dtype}")
        break

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = NeuralNetwork(X.shape[1],ndfa).to(device)
    print(model)

    alpha = .1
    if ndfa > 2:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=alpha)

    epochs = 50
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn, ndfa)
    print("Done!")

    # baseline performance
    print(summary.to_string())

    #path = os.path.join('model','model.pth')
    #torch.save(model.state_dict(), path)
    #print(f"Saved PyTorch Model State to {path}")

    # set model to evaluate mode, do not need to accumulate gradients
    model.eval()
    x, y = test_data[:][0], test_data[:][1]
    with torch.no_grad():
        pred = model(x.to(device)).cpu()
        if ndfa > 2:
            pred = pred.argmax(1)
        else:
            pred = torch.sigmoid(pred)
            pred[pred>=.5] = 1
            pred[pred<.5] = 0

    # confusion matrix - rows are true labels, columns predicted
    methods = [Ylabel[i] for i in range(len(args.dfa))]
    cm = confusion_matrix(y_test, pred)
    cmdf = pd.DataFrame(cm,index=methods,columns=methods)
    print(cmdf.to_string())

    # histogram the mean abs difference between the two methods when they are
    # right vs wrong
    ind = (pred == y).numpy()
    print('summary stats for correct guess')
    print(ARE.iloc[ind_test].iloc[ind,5:].abs().diff(axis=1).iloc[:,1].abs().describe())
    print('summary stats for incorrect guess')
    print(ARE.iloc[ind_test].iloc[~ind,5:].abs().diff(axis=1).iloc[:,1].abs().describe())
