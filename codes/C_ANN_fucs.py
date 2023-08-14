
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error
import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD
from torch.nn.modules.loss import MSELoss
import time

n2t = lambda n: torch.from_numpy(n).type(torch.float32)
def split_data(data, train_row, test_row, pred_row, pred_all_row):
    X_tr = data.iloc[:train_row, :]
    # test dataset
    X_ts = data.iloc[train_row:train_row + test_row, :]
    # prediction dataset
    X_pred = data.iloc[train_row + test_row:train_row + test_row + pred_row, :]
    # prediction all dataset
    X_pred_all = data.iloc[train_row + test_row + pred_row:train_row + test_row + pred_row + pred_all_row, :]

    X_tr = n2t(X_tr.values)
    X_ts = n2t(X_ts.values)
    X_pred = n2t(X_pred.values)
    X_pred_all = n2t(X_pred_all.values)
    return X_tr, X_ts, X_pred, X_pred_all

def r_squared(y_true, y_pred):
    # Compute the total sum of squares
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)

    # Compute the residual sum of squares
    ss_res = torch.sum((y_true - y_pred) ** 2)

    # Compute the R-squared value
    r2 = 1 - ss_res / ss_tot

    return r2
# define the function to normalize continue variables, here lon and lat
def lon_lat_norm(lon_lat,data):
    train_stats = lon_lat.describe()
    train_stats.head()
    train_stats = train_stats.transpose()
    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']
    return norm(data)

# define dataset
class DatasetWrapper_ANN_F(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
class DatasetWrapper_ANN_S_ST(Dataset):
        def __init__(self, X_1, X_2, y):
            self.X_1,self.X_2, self.y = X_1, X_2, y

        def __len__(self):
            return len(self.X_1)

        def __getitem__(self, idx):
            return self.X_1[idx],self.X_2[idx], self.y[idx]

# define models based on architectures
class MLP_ANN_F(nn.Module):
            def __init__(self, input_size_nl, hidden_sizes):
                super(MLP_ANN_F, self).__init__()
                self.layers = nn.ModuleList()
                self.layers.append(nn.Linear(input_size_nl, hidden_sizes[0]))
                self.layers.append(nn.Sigmoid())
                for i in range(1, len(hidden_sizes)):
                    self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
                    self.layers.append(nn.Sigmoid())
                self.layers.append(nn.Linear(hidden_sizes[-1], 1))

            def forward(self, x_nl):
                out = x_nl
                for layer in self.layers:
                    out = layer(out)
                return out
class MLP_ANN_S(nn.Module):
            def __init__(self, input_size_nl, hidden_sizes,input_size_l):
                super(MLP_ANN_S, self).__init__()
                self.input_size_l = input_size_l
                self.layers = nn.ModuleList()
                self.layers.append(nn.Linear(input_size_nl, hidden_sizes[0]))
                self.layers.append(nn.Sigmoid())
                for i in range(1, len(hidden_sizes)):
                    self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
                    self.layers.append(nn.Sigmoid())
                self.layers.append(nn.Linear(hidden_sizes[-1], 1))
                self.layers.append(nn.Sigmoid())
                self.concat_layer = nn.Linear(input_size_l+1, 1)

            def forward(self, x_nl,x_l):
                out = x_nl
                for layer in self.layers:
                    out = layer(out)

                out = torch.cat((out, x_l), dim=1)
                out = self.concat_layer(out)
                return out
class MLP_ANN_ST(nn.Module):
            def __init__(self, input_size_nl, hidden_sizes,input_size_l):
                super(MLP_ANN_ST, self).__init__()
                self.input_size_l = input_size_l
                self.layers = nn.ModuleList()
                self.layers.append(nn.Linear(input_size_nl, hidden_sizes[0]))
                self.layers.append(nn.Sigmoid())
                for i in range(1, len(hidden_sizes)):
                    self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
                    self.layers.append(nn.Sigmoid())
                self.layers.append(nn.Linear(hidden_sizes[-1], 1))
                self.layers.append(nn.Sigmoid())
                self.concat_layer = nn.Linear(input_size_l+1, 1)

            def forward(self, x_nl,x_l):
                out = x_nl
                for layer in self.layers:
                    out = layer(out)

                out = torch.cat((out, x_l), dim=1)
                out = self.concat_layer(out)
                return out

# training process: train_ANN_F inout 1 dataset, train_ANN_S_ST input 2 datasets. Print model, batch loss, R2, train loss, time. R2, train loss are on validation datasets (here we used entrire training set as validation set)
def train_ANN_F(net, optimizer, dataset, nepochs=100, batch_size=100, val_nl=None, val_y=None,device=None,scenario=None,NN_str=None,early_stopping=None,criterion=None):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_nl=val_nl.to(device)
    val_y=val_y.to(device)
    ## move to GPU if available
    start_time = time.time()
    for epoch in range(nepochs):
        if (epoch + 1) % 10 == 0:
            end_time = time.time()
            epoch_time = end_time - start_time
            print(
                'Scenario {}, {}, Epoch [{}/{}], Batch_Loss: {:.4f},R2: {:.4f},Total_Loss: {:.4f}, time: {:.4f}'.format(
                    scenario,
                    NN_str,
                    epoch + 1,
                    nepochs,
                    loss.item(),
                    r_squared_val,
                    val_loss,
                    epoch_time))
            start_time = time.time()
        for tr_nl, labels in dataloader:
            tr_nl = tr_nl.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            outputs = net(tr_nl.float())
            loss = criterion(outputs.squeeze(), labels.float().squeeze())

            # backward pass and optimization
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            val_out = torch.exp(net(val_nl.float()))
            val_loss = criterion(val_out.squeeze(), val_y.float().squeeze())
            r_squared_val = r_squared(val_y.float().squeeze(), val_out.squeeze())
        early_stopping(val_loss, net)
        if early_stopping.early_stop:
            print('Early stopping')
            break
def train_ANN_S_ST(net, optimizer, loss, dataset, nepochs=100, batch_size=100, val_nl=None, val_l=None,val_y=None,device=None,scenario=None,NN_str=None,early_stopping=None,criterion=None):
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                val_nl = val_nl.to(device)
                val_l = val_l.to(device)
                val_y = val_y.to(device)
                ## move to GPU if available
                start_time = time.time()
                for epoch in range(nepochs):
                    if (epoch + 1) % 10 == 0:
                        end_time = time.time()
                        epoch_time = end_time - start_time
                        print(
                            'Scenario {}, {}, Epoch [{}/{}], Batch_Loss: {:.4f},R2: {:.4f},Total_Loss: {:.4f}, time: {:.4f}'.format(
                                scenario,
                                NN_str,
                                epoch + 1,
                                nepochs,
                                loss.item(),
                                r_squared_val,
                                val_loss,
                                epoch_time))
                        start_time = time.time()
                    for tr_nl, tr_l, labels in dataloader:
                        tr_nl = tr_nl.to(device)
                        tr_l = tr_l.to(device)
                        labels = labels.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward pass
                        outputs = net(tr_nl.float(), tr_l.float())
                        loss = criterion(outputs.squeeze(), labels.float().squeeze())

                        # backward pass and optimization
                        loss.backward()
                        optimizer.step()
                    with torch.no_grad():
                        val_out = torch.exp(net(val_nl.float(), val_l.float()))
                        val_loss = criterion(val_out.squeeze(), val_y.float().squeeze())
                        r_squared_val = r_squared(val_y.float().squeeze(), val_out.squeeze())
                    early_stopping(val_loss, net)
                    if early_stopping.early_stop:
                        print('Early stopping')
                        break

# early stopping for all architectures
class EarlyStopping:
    def __init__(self, patience=10, delta=0, verbose=True, loc=None,NN_str=None):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.verbose = verbose
        self.loc = loc
        self.NN_str=str(NN_str)

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if val_loss < self.val_loss_min:
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            model_save_path = os.path.join(self.loc, "model/")
            if not os.path.exists(os.path.dirname(model_save_path)):
                os.makedirs(os.path.dirname(model_save_path))
            torch.save(model.state_dict(), model_save_path + self.NN_str + '.pt')
            self.val_loss_min = val_loss
