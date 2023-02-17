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

from utils import get_data, get_pop_data

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

def EE_forward(net1, net2, x):
    x.requires_grad = True
    f1 = net1(x)
    net1.zero_grad()
    f1.backward()
    dc = torch.cat([x.grad.data.detach(), x.detach()], dim=1)
    dc = dc / torch.linalg.norm(dc)
    f2 = net2(dc)
    return f1.item(), f2.item(), dc

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

def test(net1, net2, X, y):
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    num = X.size(0)
    acc = 0.0
    ci = torch.zeros(1, X.shape[1]).to(device)

    for i in range(num):
        x, y = dataset[i]
        x = x.view(1, -1).to(device)
        x0 = torch.cat([x, ci], dim=1)
        x1 = torch.cat([ci, x], dim=1)
        u0 = net1(x0)
        u1 = net1(x1)
        
        lbl = y.item()
        if u0 > u1:
            pred = 0
        else:
            pred = 1
        if pred == lbl:
            acc += 1
    
    print("Test Acc:{:.2f}".format(acc * 100.0 / num))

def get_ber(label_ratio):
    threshold_value = 1.1
    if label_ratio == 0.2:
        if dataset_name in ['fashion']:
            threshold_value = 0.85
        if dataset_name in ['phishing', 'ijcnn']:
            threshold_value = 0.9
        elif dataset_name in ['mnist']:
            threshold_value = 0.95
    elif label_ratio == 0.5:
        if dataset_name in ['phishing', 'ijcnn', 'fashion']:
            threshold_value = 0.6
        elif dataset_name in ['mnist']:
            threshold_value = 0.65
    return threshold_value
    
label_ratio = 0.03

device = 'cuda'
test_mode = 'regret'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# [phishing, ijcnn, letter, fashion, mnist, cifar]
dataset_name = 'phishing'
if dataset_name in ['mnist', 'phishing']:
    margin = 6
else:
    margin = 7

print(dataset_name, label_ratio)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

if __name__ == "__main__":
    if test_mode == 'regret':
        X, Y = get_data(dataset_name)
    elif test_mode == 'accuracy':
        X, Y, test_X, test_Y = get_pop_data(dataset_name)
    
    dataset = TensorDataset(torch.tensor(X.astype(np.float32)), torch.tensor(Y.astype(np.int64)))

    net1 = Network_exploitation(X.shape[1] * 2).to(device)
    net2 = Network_exploration(X.shape[1] * 2 * 2).to(device)
    regret = []
    X1_train, X2_train, y1, y2 = [], [], [], []
    n = len(dataset)
    budget = int(n * label_ratio)
    current_regret = 0.0
    query_num = 0
    tf = time.time()
    ci = torch.zeros(1, X.shape[1]).to(device)

    if label_ratio <= 0.1:
        ber = 1.1
    else:
        ber = get_ber(label_ratio)

    for i in range(n):
        x, y = dataset[i]
        x = x.view(1, -1).to(device)
        x0 = torch.cat([x, ci], dim=1)
        x1 = torch.cat([ci, x], dim=1)

        f1_0, f2_0, dc_0 = EE_forward(net1, net2, x0)
        f1_1, f2_1, dc_1 = EE_forward(net1, net2, x1)
        u0 = f1_0 + 1 / (i+1) * f2_0
        u1 = f1_1 + 1 / (i+1) * f2_1

        ind = 0
        if u0 > u1:
            pred = 0
            if u0 - u1 < margin * 0.1:
                ind = 1
        else:
            pred = 1
            if u1 - u0 < margin * 0.1:
                ind = 1

        lbl = y.item()
        if pred != lbl:
            current_regret += 1
            reward = 0
        else:
            reward = 1

        if not ind and query_num < budget:
            if random.random() > ber:
                ind = 1

        if ind and (query_num < budget): 
            query_num += 1
            if pred == 0:
                X1_train.append(x0)
                X2_train.append(dc_0)
                y1.append(torch.Tensor([reward]))
                y2.append(torch.Tensor([reward - f1_0 - f2_0]))

                X1_train.append(x1)
                X2_train.append(dc_1)
                y1.append(torch.Tensor([1 - reward]))
                y2.append(torch.Tensor([1 - reward - f1_1 - f2_1]))
            else:
                X1_train.append(x1)
                X2_train.append(dc_1)
                y1.append(torch.Tensor([reward]))
                y2.append(torch.Tensor([reward - f1_1 - f2_1]))

                X1_train.append(x0)
                X2_train.append(dc_0)
                y1.append(torch.Tensor([1 - reward]))
                y2.append(torch.Tensor([1 - reward - f1_0 - f2_0]))

            train_NN_batch(net1, X1_train, y1)
            train_NN_batch(net2, X2_train, y2)

        # if (i+1) % 1000 == 0:
        #     print("Time:{:.2f}\tIters:{}\tRegret:{:.1f}".format(time.time()-tf, i+1, current_regret))
        #     tf = time.time()
        regret.append(current_regret)
       
    print(query_num)
    if test_mode == 'regret':
        print(current_regret)
    else:
        test_X, test_Y = torch.tensor(test_X.astype(np.float32)), torch.tensor(test_Y.astype(np.int64))
        test(net1, net2, test_X, test_Y)




