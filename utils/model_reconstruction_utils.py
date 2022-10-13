import re

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class StaticReprDataset(Dataset):
    def __init__(self, X, Y):
        self.x = X
        self.y = Y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])

def get_pers_repr_generic(i, dir, device, fits=10, only = None):
    model = torch.load(dir + str(i) + '_' + str(fits) + '.pth', map_location=device)

    tmp = {}
    for n, p in model.named_parameters():
        if only != None:
            # ignores parameters that are not in 'only'
            if only not in n:
                continue

        if len(p.shape) > 1:
            tmp[n] = np.squeeze(p.cpu().detach().numpy()).flatten()
        else:
            tmp[n] = p.cpu().detach().numpy()


    return np.hstack(list(tmp.values()))

def get_best_seed(path = './results.md'):
    with open(path) as f:
        lines = f.readlines()

    for l in lines:
        if 'best seed' in l:
            l_split = l.split('seed')
            return int(l_split[-1].strip())

def get_best_seed_bl(path = './results.md'):
    with open(path) as f:
        lines = f.readlines()

    for l in lines:
        if 'best baseline seed' in l:
            l_split = l.split('seed')
            return int(l_split[-1].strip())

def overwrite_model_weights_generic(model, repr, dtype, only=None):
    index_one = 0
    index_two = 0

    for n, p in model.named_parameters():
        if only != None:
            # ignores parameters that are not in 'only'
            if only not in n:
                continue

        index_two += np.squeeze(p.cpu().detach().numpy()).flatten().shape[0]
        p.data = torch.nn.Parameter(repr.to(dtype)[:, index_one:index_two].reshape(p.shape))
        index_one = index_two

    return model

def get_pers_repr(i, dir, device, fits=10):
    model = torch.load(dir + str(i) + '_' + str(fits) + '.pth', map_location=device)
    b1 = model.module.linear1.bias.cpu().detach().numpy()
    b2 = model.module.linear2.bias.cpu().detach().numpy()
    # b3 = model.module.hidden3.bias.cpu().detach().numpy()

    return np.hstack((b1, b2))


def overwrite_model_weights(model, repr, dtype):
    b1 = repr.to(dtype)[:, :model.linear1.bias.shape[0]]
    b2 = repr.to(dtype)[:,
         model.linear1.bias.shape[0]:model.linear1.bias.shape[0] + model.linear2.bias.shape[0]]
    # b3 = repr.to(params['dtype'])[:, model.hidden1.bias.shape[0] + model.hidden2.bias.shape[0]:]
    model.linear1.bias = torch.nn.Parameter(b1)
    model.linear2.bias = torch.nn.Parameter(b2)
    # model.hidden3.bias = torch.nn.Parameter(b3)

    return model
    # return np.hstack((b1,b2,b3))


def evaluate(encoder, dataloader, device):
    # encoder.eval()
    mae = 0.
    rmse = 0.

    mae_fn = nn.L1Loss()
    mse_fn = nn.MSELoss()
    with torch.no_grad():
        for iter, batch in enumerate(dataloader):
            inputs, targets = batch[0].to(device), batch[1].to(device)
            preds = encoder(inputs.to(device))

            # mae += torch.abs(preds.flatten() - targets.flatten()).sum().data
            mae += mae_fn(targets, preds)
            # rmse += ((preds.flatten() - targets.flatten()) ** 2).sum().data
            rmse += torch.sqrt(mse_fn(targets, preds))
    # mae = mae  # / len(dataloader)
    # rmse = rmse  # / len(dataloader)

    print('MAE: {}'.format(mae))
    print('RMSE: {}'.format(rmse))

    return rmse, mae


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim=64, output_dim=3):
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden3 = nn.Linear(hidden_dim, output_dim)
        self.fn1 = torch.nn.GELU()
        self.fn2 = torch.nn.GELU()
        self.fn3 = torch.nn.GELU()

    def forward(self, input):
        x = self.fn1(self.hidden1(input))
        x = self.fn2(self.hidden2(x))
        x = self.fn3(self.hidden3(x))
        return x


def infos_from_dirs(dirs, sa, ds):
    summary = []
    for dir in dirs:
        if ds not in dir:
            continue
        n = int(re.search(r'test-\d+-maml', dir).group().split('-')[1])
        steps_ahead = int(re.search('(bp_\d+|o2_\d+|si_\d+)', dir).group()[3:])
        summary.append((dir, steps_ahead, n))

    filtered_summary = [x for x in summary if x[1] == sa]
    assert (len(filtered_summary) == 1)

    return filtered_summary[0]


def euclidean_distance(x, y):
    return np.sqrt(np.sum(np.power(x - y, 2)))


def nearest_neighbour(X, all_features):
    min_dist = 10000
    index = -1
    for i in range(len(X)):
        curr_dist = euclidean_distance(X[i], all_features)
        if curr_dist == 0.:
            print('min_distance: ', 0.)
            return i
        if curr_dist < min_dist:
            min_dist = curr_dist
            index = i
    print('min_distance: ', min_dist)
    return index

class SineEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim=64, output_dim=3):
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden3 = nn.Linear(hidden_dim, output_dim)


    def forward(self, input):
        x = torch.tanh(self.hidden1(input))
        x = torch.tanh(self.hidden2(x))
        x = self.hidden3(x)
        return x

# DECREASE 1.45/1.6 with 6 layers
class SineEncoder2(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=3):
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden3 = nn.Linear(hidden_dim, output_dim)
        self.dropout1 = nn.Dropout(p=0.1)


    def forward(self, input):
        x = self.dropout1(torch.tanh(self.hidden1(input)))
        x = torch.tanh(self.hidden2(x))
        x = self.hidden3(x)
        return x


class SineEncoder3(nn.Module):

    def __init__(self, input_dim, hidden_dim=64, output_dim=3):
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, output_dim)
        self.fn = nn.ReLU()


    def forward(self, input):
        x = self.fn(self.hidden1(input))
        x = self.hidden2(x)
        return x

class SXEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim=64, output_dim=3):
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden3 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden4 = nn.Linear(hidden_dim, hidden_dim)


        self.last = nn.Linear(hidden_dim, output_dim)
        self.fn1 = nn.ReLU()
        self.fn2 = nn.ReLU()
        self.fn3 = nn.ReLU()
        self.fn4 = nn.ReLU()


    def forward(self, input):
        x = self.fn1(self.hidden1(input))
        x = self.fn2(self.hidden2(x))
        x = self.fn3(self.hidden3(x))
        x = self.fn4(self.hidden4(x))

        x = self.last(x)
        return x