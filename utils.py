import arff
import pickle
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def read_data_arff(file_path, dataset):
	data = arff.load(open(file_path, 'r'))
	data = data['data']
	n, m = len(data), len(data[0])
	X, Y = np.zeros([n, m-1]), np.zeros([n])
	if dataset == 'ijcnn':
		for i in range(n):
			entry = data[i]
			if float(entry[-1]) == -1:
				Y[i] = 0
			elif float(entry[-1]) == 1:
				Y[i] = 1
			else:
				raise ValueError
			for j in range(m-1):
				X[i, j] = float(entry[j])

	return X, Y

def read_data_txt(file_path, dataset):
	f = open(file_path, "r").readlines()
	n = len(f)
	if dataset == 'phishing':
		m = 68
		X = np.zeros([n, 68])
		Y = np.zeros([n])
		for i, line in enumerate(f):
			line = line.strip().split()
			lbl = int(line[0])
			if lbl != 0 and lbl != 1:
				raise ValueError
			Y[i] = lbl
			l = len(line)
			for item in range(1, l):
				pos, value = line[item].split(':')
				pos, value = int(pos), float(value)
				X[i, pos-1] = value

	return X, Y

def load_data(dataset):
	if dataset in ['ijcnn']:
		file_path = './binary_data/{}.arff'.format(dataset)
		return read_data_arff(file_path, dataset)
	
	elif dataset in ['phishing']:
		file_path = './binary_data/{}.txt'.format(dataset)
		return read_data_txt(file_path, dataset)
	
	elif dataset in ['letter', 'fashion']:
		file_path = './binary_data/{}_binary_data.pt'.format(dataset)
		f = open(file_path, 'rb')
		data = pickle.load(f)
		X, Y = data['X'], data['Y']
		return X, Y
	
	elif dataset in ['mnist']:
		file_path = './MNIST_data/MNIST_binary_data.pt'
		f = open(file_path, 'rb')
		data = pickle.load(f)
		X, Y = data['X'], data['Y']
		return X, Y
	
	elif dataset in ['cifar']:
		file_path = './CIFAR10_data/CIFAR10_binary_data.pt'
		f = open(file_path, 'rb')
		data = pickle.load(f)
		X, Y = data['X'], data['Y']
		return X, Y

def get_data(dataset_name):
	X, Y = load_data(dataset_name)
	index = np.arange(X.shape[0])
	np.random.shuffle(index)
	if X.shape[0] > 12000:
		index = index[:12000]
	print(index[:5])
	X = X[index, :]
	Y = Y[index]
	return X, Y

def get_pop_data(dataset_name):
	X, Y = load_data(dataset_name)
	index = np.arange(X.shape[0])
	np.random.shuffle(index)
	if X.shape[0] > 12000:
		index = index[:12000]
	print(index[:5])
	online_X = X[index[:10000], :]
	online_Y = Y[index[:10000]]
	test_X = X[index[10000:], :]
	test_Y = Y[index[10000:]]
	return online_X, online_Y, test_X, test_Y

def get_pretrain(dataset_name, label_ratio, num_sample):
	X, Y = load_data(dataset_name)
	index = np.arange(X.shape[0])
	np.random.seed(42)
	np.random.shuffle(index)
	if X.shape[0] > 12000:
		index = index[:12000]
	
	X = X[index, :]
	Y = Y[index]
	n = X.shape[0]
	if label_ratio > 0.1:
		bernoulli_prob = 0.5
	else:
		bernoulli_prob = 0.9

	pre_X, pre_Y = np.zeros([num_sample, X.shape[1]]), np.zeros([num_sample])
	num = 0
	for i in range(n):
		q = random.random()
		if q > bernoulli_prob and num < num_sample:
			pre_X[num, :] = X[i, :]
			pre_Y[num] = Y[i]
			num += 1

	return pre_X, pre_Y

if __name__ == "__main__":
	dataset = 'mnist'
	
	random.seed(42)
	np.random.seed(42)
	X, Y = get_data(dataset)
	print(X.shape, Y.shape)

	

