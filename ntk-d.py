import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils import get_data, get_pop_data

def neural_forward(model, x, Z):
    p = model(x)
    model.zero_grad()
    p.backward()
    g = torch.cat([p.grad.flatten().detach() for p in model.parameters()])
    sigma = gamma * g * g / Z
    sigma = torch.sqrt(torch.sum(sigma))
    u = p.item() + sigma.item()

    return u, g, sigma.item()

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

def test(model, X, y, Z):
    dataset = TensorDataset(X, y)
    num = X.size(0)
    acc = 0.0
    ci = torch.zeros(1, X.shape[1]).to(device)

    for i in range(num):
        x, y = dataset[i]
        x = x.view(1, -1).to(device)
        x0 = torch.cat([x, ci], dim=1)
        x1 = torch.cat([ci, x], dim=1)
        # u0, __, __ = neural_forward(model, x0, Z)
        # u1, __, __ = neural_forward(model, x1, Z)
        u0 = model(x0)
        u1 = model(x1)
        lbl = y.item()
        if u0 > u1:
            pred = 0
        else:
            pred = 1
        if pred == lbl:
            acc += 1
    
    print("Test Acc:{:.2f}".format(acc * 100.0 / num))

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size=100):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

label_ratio = 0.03
test_mode = 'regret'
lam = 1.0
gamma = 0.1
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# [phishing, ijcnn, letter, fashion, mnist, cifar]
dataset_name = 'cifar'
print(dataset_name, label_ratio)

if __name__ == "__main__":
    if test_mode == 'regret':
        X, Y = get_data(dataset_name)
    elif test_mode == 'accuracy':
        X, Y, test_X, test_Y = get_pop_data(dataset_name)
    dataset = TensorDataset(torch.tensor(X.astype(np.float32)), torch.tensor(Y.astype(np.int64)))

    model = MLP(X.shape[1] * 2).to(device)
    regret = []
    X_train, reward = [], []
    n = len(dataset)
    budget = int(n * label_ratio)
    current_regret = 0.0
    query_num = 0
    tf = time.time()

    total_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Z = lam * torch.ones((total_param)).to(device)
    ci = torch.zeros(1, X.shape[1]).to(device)

    for i in range(n):
        x, y = dataset[i]
        x = x.view(1, -1).to(device)
        x0 = torch.cat([x, ci], dim=1)
        x1 = torch.cat([ci, x], dim=1)

        u0, g0, sigma0 = neural_forward(model, x0, Z)
        u1, g1, sigma1 = neural_forward(model, x1, Z)

        ind = 0
        lt = 0
        if u0 > u1:
            pred = 0
            B = 2 * sigma0
            if abs(u0 - 0.5) <= B:
                ind = 1
        else:
            pred = 1
            B = 2 * sigma1
            if abs(u1 - 0.5) <= B:
                ind = 1

        lbl = y.item()
        if pred != lbl:
            current_regret += 1
            lt = 1

        if ind and (query_num < budget): 
            query_num += 1
            if pred == 0:
                Z += g0 * g0
                X_train.append(x0)
            else:
                Z += g1 * g1
                X_train.append(x1)

            reward.append(torch.Tensor([1-lt]))
            train_NN_batch(model, X_train, reward)
            
        # if (i+1) % 1000 == 0:
        #     print("Time:{:.2f}\tIters:{}\tRegret:{:.1f}".format(time.time()-tf, i+1, current_regret))
        #     tf = time.time()
        regret.append(current_regret)
       
    print(query_num)
    if test_mode == 'regret':
        print(current_regret)
        # np.save('./res/{}/neuralucb_res.npy'.format(dataset_name), regret)
    else:
        test_X, test_Y = torch.tensor(test_X.astype(np.float32)), torch.tensor(test_Y.astype(np.int64))
        test(model, test_X, test_Y, Z)


