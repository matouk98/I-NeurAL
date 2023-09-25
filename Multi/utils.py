import sys
import arff
import pickle
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        if self.stdout is None:
            print("Fuck!")
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        if not '...' in data:
            self.file.write(data)
        self.stdout.write(data)
        self.flush()
    def flush(self):
        self.file.flush()

def load_data(dataset):
	file_path = './multi_data/{}_multi_data.pt'.format(dataset)
	f = open(file_path, 'rb')
	data = pickle.load(f)
	X, Y = data['X'], data['Y']
	return X, Y
	
def get_data(dataset_name):
	X, Y = load_data(dataset_name)
	index = np.arange(X.shape[0])
	np.random.shuffle(index)
	if X.shape[0] > 10000:
		index = index[:10000]
	print(index[:5])
	X = X[index, :]
	Y = Y[index]
	return X, Y

def get_pretrain(dataset_name, num_sample=360):
	X, Y = load_data(dataset_name)
	index = np.arange(X.shape[0])
	np.random.seed(42)
	np.random.shuffle(index)
	if X.shape[0] > 12000:
		index = index[:12000]
	
	X = X[index, :]
	Y = Y[index]
	n = X.shape[0]

	pre_X, pre_Y = np.zeros([num_sample, X.shape[1]]), np.zeros([num_sample])
	num = 0
	for i in range(n):
		q = random.random()
		if q > 0.9 and num < num_sample:
			pre_X[num, :] = X[i, :]
			pre_Y[num] = Y[i]
			num += 1

	return pre_X, pre_Y

if __name__ == "__main__":
	dataset = 'MNIST'
	
	random.seed(42)
	np.random.seed(42)
	X, Y = get_data(dataset)
	print(X.shape, Y.shape)

	

