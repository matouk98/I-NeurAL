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

def train_cls_batch(model, X, y, num_epochs=10, lr=0.001, batch_size=64):
    model.train()
    X = torch.cat(X).float()
    y = torch.cat(y).float()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_fn = nn.BCELoss().to(device)
    num = X.size(0)

    for i in range(num_epochs):
        batch_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x).view(-1)

            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
        
        if batch_loss / num <= 1e-3:
            return batch_loss / num

    return batch_loss / num

def test(model, X, y):
    model.eval()
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    num = X.size(0)
    acc = 0.0
    # print(num)
    
    for x,y in dataloader:
        x, y = x.to(device), y.to(device)
        pred = model(x).view(-1)
        pred = (pred > 0.5)
        acc += torch.sum(pred == y).item()
    
    print("Test Acc:{:.2f}".format(acc * 100.0 / num))

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size=100):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc2(self.activate(self.fc1(x))))

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

label_ratio = 0.03
device = 'cuda'
test_mode = 'regret'
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
    model = MLP(X.shape[1]).to(device)
    regret = []
    X_train, y_train = [], []
    n = len(dataset)
    budget = int(n * label_ratio)
    if label_ratio == 0.03:
        ber = 0.9
    elif label_ratio == 0.10:
        ber = 0.85
    elif label_ratio == 0.20:
        ber = 0.75
    elif label_ratio == 0.50:
        ber = 0.45

    current_regret = 0.0
    query_num = 0
    tf = time.time()

    for i in range(n):
        model.eval()
        x, y = dataset[i]
        x = x.view(1, -1).to(device)
        prob = model(x).item()
        pred = int(prob >= 0.5)
        lbl = y.item()
        if pred != lbl:
            current_regret += 1

        q = random.random()
        
        if q >= ber and (query_num < budget):
            X_train.append(x)
            y_train.append(y.view(-1))
            loss = train_cls_batch(model, X_train, y_train)
            query_num += 1

        # if (i+1) % 1000 == 0:
        #     print("Time:{:.2f}\tIters:{}\tRegret:{:.1f}".format(time.time()-tf, i+1, current_regret))
        #     tf = time.time()

        regret.append(current_regret)
        
    print(query_num)
    if test_mode == 'regret':
        print(current_regret)
        # np.save('./res/{}/neural_res.npy'.format(dataset_name), regret)
    else:
        test_X, test_Y = torch.tensor(test_X.astype(np.float32)), torch.tensor(test_Y.astype(np.int64))
        test(model, test_X, test_Y)


