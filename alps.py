import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils import get_data, get_pop_data, get_pretrain

def train_cls_batch(model, X, y, num_epochs=10, lr=0.001, batch_size=64):
    model.train()
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()

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

def learn(H, S, T, cur_label=None):
    min_loss, H_hat = 1e9, None
    for j in range(num_model):
        if model_info[j]['consistent'] == False:
            continue
        if cur_label is not None:
            prob = pred_now[j][0]
            pred = int(prob >= 0.5)
            if pred != cur_label:
                continue

        tot_loss = model_info[j]['sum_loss']
        if len(S) + len(T) > 0:
            tot_loss /= (len(T) + len(S))
        if tot_loss < min_loss:
            min_loss = tot_loss
            H_hat = model

    return H_hat, min_loss

def shrink(p_list, set_T, F_class):
    loss_list, new_F_class = [], []
    
    for j, score in F_class:
        loss_tot = F_class_info[(j, score)]['sum_loss']
        loss_tot /= len(set_T)
        loss_list.append(loss_tot)

    min_loss = min(loss_list)
    for i, (j, score) in enumerate(F_class):
        if loss_list[i] <= min_loss + delta1:
            new_F_class.append((j, score))
    return new_F_class

def calc_p(F_class, y):
    loss_list = []
    
    for j, score in F_class:
        prob, loss0, loss1 = pred_now[j]
        # requester function return 0
        if abs(prob - 0.5) * 2 >= score:
            if y == 0:
                loss = loss0 
            else:
                loss = loss1
        else:
            loss = 0
        loss_list.append(loss)

    return max(loss_list) - min(loss_list)

def calc_r(S, yhat):
    mini_score, rn = 1.0, 0
    for j, score in F_class:
        if model_info[j]['consistent']:
            prob = pred_now[j][0]
            pred = int(prob >= 0.5)
            if pred != yhat:
                continue
            if score < mini_score:
                margin = abs(prob - 0.5) * 2
                if margin >= score:
                    rn = 0
                else:
                    rn = 1
    return rn

def update_xn(H_class, xn):
    global pred_now
    pred_now = []
    loss_fn = nn.BCELoss().to(device)
    with torch.no_grad():
        for model in H_class:
            prob = model(xn).view(-1)
            l0 = loss_fn(prob, label0).item()
            l1 = loss_fn(prob, label1).item()
            pred_now.append((prob.item(), l0, l1))

def update_set(H_class, F_class, p, yn, flag='S'):
    for i, model in enumerate(H_class):
        prob = pred_now[i][0]
        if yn == 0:
            loss = pred_now[i][1]
        else:
            loss = pred_now[i][2]
        if flag == 'S':
            pred = int(prob >= 0.5)
            if pred != yn:
                model_info[i]['consistent'] = False
        model_info[i]['sum_loss'] += loss
    
    if p != 0:
        for j, score in F_class:
            prob = pred_now[j][0]
            if yn == 0:
                loss = pred_now[j][1]
            else:
                loss = pred_now[j][2]
            # not request
            if abs(prob - 0.5) * 2 >= score:
                F_class_info[(j, score)]['sum_loss'] += loss * p


def test_model_accuracy(H_class, X, y):
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()

    num = X.size(0)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    for model in H_class:
        acc = 0.0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                pred = model(x).view(-1)
                pred = pred >= 0.5
                acc += torch.sum(pred == y)

        print("Acc:{:.2f}".format(acc * 100.0 / num))

def test_model_margin(H_class, dataset):
    for model in H_class:
        with torch.no_grad():
            tot_margin = 0.0
            for i in range(100):
                xn, yn = dataset[i]
                xn = xn.view(1, -1).to(device)
                pred = model(xn).item()
                tot_margin += abs(pred - 0.5) * 2
            
            print("Margin:{:.2f}".format(tot_margin / 100))

def test(model, X, y):
    model.eval()
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    num = X.size(0)
    acc = 0.0
    
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
if label_ratio <= 0.10:
    num_model = 20
else:
    num_model = 5
delta1, delta2 = 0.5, 0.5
device = 'cuda'
test_mode = 'regret'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# [phishing, ijcnn, letter, fashion, mnist, cifar]
dataset_name = 'cifar'

# For pred_now, we store a tuple (prob, loss0, loss1)
pred_now = []
model_info, F_class_info = {}, {}
label0, label1 = torch.Tensor([0]).to(device), torch.Tensor([1]).to(device)

print(dataset_name, label_ratio)

if __name__ == "__main__":
    if test_mode == 'regret':
        X, Y = get_data(dataset_name)
    elif test_mode == 'accuracy':
        X, Y, test_X, test_Y = get_pop_data(dataset_name)
    dataset = TensorDataset(torch.tensor(X.astype(np.float32)), torch.tensor(Y.astype(np.int64)))

    regret = []
    # [x, y (scalar)]
    set_S, set_T = [], []
    n = len(dataset)
    budget = int(n * label_ratio)
    query_num = 0
    pre_X, pre_Y = get_pretrain(dataset_name, label_ratio, budget)
    print(pre_X.shape[0])
    tf = time.time()
    F_class, H_class = [], []
    for i in range(num_model):
        torch.manual_seed(42+i)
        model = MLP(X.shape[1]).to(device)
        for k in range(budget):
            train_cls_batch(model, pre_X[:k+1, :], pre_Y[:k+1])
        
        model = model.eval()
        for s in [0.1, 0.25, 0.5, 0.75, 0.9]:
            F_class.append((i, s))
            F_class_info[(i,s)] = {}
            F_class_info[(i,s)]['sum_loss'] = 0.0
        
        H_class.append(model)
        model_info[i] = {}
        model_info[i]['sum_loss'] = 0.0
        model_info[i]['consistent'] = True
    
    # print("Time:{:.2f}".format(time.time()-tf))
    # exit()
    # test_model_accuracy(H_class, pre_X, pre_Y)
    # test_model_margin(H_class, dataset)
    
    current_regret = 0.0
    p_list = []
    tf = time.time()
    time_cost = 0.0

    for i in range(n):
        xn, yn = dataset[i]
        xn = xn.view(1, -1).to(device)
        yn = yn.view(-1).float().to(device)
        if i == 0:
            hn = H_class[0]
        else:
            hn = learn(H_class, set_S, set_T)[0]

        with torch.no_grad():
            prob = hn(xn).item()
            pred = int(prob >= 0.5)
            lbl = yn.item()
            if pred != lbl:
                current_regret += 1
        update_xn(H_class, xn)
        
        h0, err0 = learn(H_class, set_S, set_T, 0)
        h1, err1 = learn(H_class, set_S, set_T, 1)
        assert h0 is not None or h1 is not None
        
        if len(set_T) > 0:
            F_class = shrink(p_list, set_T, F_class)
        
        p = max(calc_p(F_class, 0), calc_p(F_class, 1))
        p = min(p, 1)
        Q = torch.bernoulli(torch.Tensor([p]))
        if p == 0:
            p = Q.item() * 1.0 * 10000
        else:
            p = Q.item() * 1.0 / p
        
        # emp_err0, emp_err1 = calc_emp_error(h0, set_S + set_T), calc_emp_error(h1, set_S + set_T)
        if (h0 is None or err0 - err1 > delta2) and Q == 0:
            rn = calc_r(set_S, 1)
            if rn == 1:
                query_num += 1
                set_T.append(yn.item())
                update_flag, update_y = 'T', yn.item()
                p_list.append(p)
            else:
                set_S.append(1)
                update_flag, update_y = 'S', 1

        elif (h1 is None or err1 - err0 > delta2) and Q == 0:
            rn = calc_r(set_S, 0)
            if rn == 1:
                query_num += 1
                set_T.append(yn.item())
                update_flag, update_y = 'T', yn.item()
                p_list.append(p)
            else:
                set_S.append(0)
                update_flag, update_y = 'S', 0
        else:
            query_num += 1
            set_T.append(yn.item())
            update_flag, update_y = 'T', yn.item()
            p_list.append(p)

        update_set(H_class, F_class, p, update_y, update_flag)
        
        # if (i+1) % 1000 == 0:
        #     print("Time:{:.2f}\tIters:{}\tRegret:{:.1f}".format(time.time()-tf, i+1, current_regret))
        #     tf = time.time()

        regret.append(current_regret)

    print(query_num)
    if test_mode == 'regret':
        print(current_regret)
    else:
        test_X, test_Y = torch.tensor(test_X.astype(np.float32)), torch.tensor(test_Y.astype(np.int64))
        test(model, test_X, test_Y)
    
