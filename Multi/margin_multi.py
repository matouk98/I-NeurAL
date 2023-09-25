import os
import time
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils import get_data, Tee

class Network_exploitation(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network_exploitation, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))
    
    
class Network_exploration(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network_exploration, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))

def EE_forward(net1, x):
    f1 = net1(x)
    return f1.item()

def train_NN_batch(model, X, y, num_epochs=10, lr=0.001, batch_size=64):
    model.train()
    X = torch.cat(X).float()
    y = torch.cat(y).float()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num = X.size(0)

    for i in range(num_epochs):
        batch_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x).view(-1)

            loss = torch.mean((pred - y) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
        
        if batch_loss / num <= 1e-3:
            return batch_loss / num

    return batch_loss / num

dataset_name = 'MNIST'
margin = 0.4

num_cls = 10
if dataset_name == 'Letter':
    num_cls = 26

device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

if __name__ == "__main__":
    Tee('{}_margin.txt'.format(dataset_name), 'w')
    X, Y = get_data(dataset_name)
    dataset = TensorDataset(torch.tensor(X.astype(np.float32)), torch.tensor(Y.astype(np.int64)))
    zero_dim = X.shape[1]

    print(dataset_name, margin, zero_dim)
    input_dim = X.shape[1] + (num_cls-1) * zero_dim
    net1 = Network_exploitation(input_dim).to(device)
    net2 = Network_exploration(input_dim * 2).to(device)

    regret = []
    X1_train, X2_train, y1, y2 = [], [], [], []
    n = len(dataset)
    budget = int(n * 0.05)
    current_regret = 0.0
    query_num = 0
    tf = time.time()

    ci = torch.zeros(1, zero_dim).to(device)
    for i in range(n):
        x, y = dataset[i]
        x = x.view(1, -1).to(device)
        x_list = []
        for k in range(num_cls):
            inputs = []
            for l in range(k):
                inputs.append(ci)
            inputs.append(x)
            for l in range(k+1, num_cls):
                inputs.append(ci)
            inputs = torch.cat(inputs, dim=1)
            x_list.append(inputs)

        f1_list, u_list = [], []
        prob = -1
        for k in range(num_cls):
            f1_k = EE_forward(net1, x_list[k])
            u_k = f1_k
            f1_list.append(f1_k)
            u_list.append((k, u_k))
            if u_k > prob:
                prob = u_k
                pred = k

        u_list = sorted(u_list, key=lambda x: x[1], reverse=True)
        # assert pred == u_list[0][0] and u_list[0][1] == prob
        pred = u_list[0][0]
        diff = u_list[0][1] - u_list[1][1]
        if diff < margin:
            ind = 1
        else:
            ind = 0
       
        lbl = y.item()
        if pred != lbl:
            current_regret += 1
            reward = 0
        else:
            reward = 1

        if ind and (query_num < budget): 
            query_num += 1
            for k in range(num_cls):
                if k != pred and k != lbl:
                    continue
                X1_train.append(x_list[k].detach().cpu())
                if k == pred:
                    y1.append(torch.Tensor([reward]))
                else:
                    y1.append(torch.Tensor([1 - reward]))
                    
            train_NN_batch(net1, X1_train, y1)
            

        if (i+1) % 1000 == 0:
            print("Time:{:.2f}\tIters:{}\tRegret:{:.1f}".format(time.time()-tf, i+1, current_regret))
            tf = time.time()
        regret.append(current_regret)
       
    print(query_num, current_regret)
    np.save('./res/{}/margin_res.npy'.format(dataset_name), regret)

