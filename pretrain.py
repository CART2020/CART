import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import pickle
import numpy as np
from torch.nn import functional as F
import time
from torch.autograd import gradcheck
import argparse

from pn import PolicyNetwork
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import glob
import os
from time import sleep

random.seed(1)
torch.cuda.manual_seed_all(1)
torch.manual_seed(1)


def cuda_(var):
    return var.cuda() if torch.cuda.is_available() else var


def validate(purpose, train_list, valid_list, test_list, model):
    model.eval()

    if purpose == 1:
        data = train_list#
    elif purpose == 2:
        data = valid_list
    else:
        data = test_list

    data = data
    bs = 64
    max_iter = int(len(data) / bs)
    start = time.time()
    epoch_loss = 0
    correct = 0

    y_true, y_pred = list(), list()
    for iter_ in range(max_iter):
        left, right = iter_ * bs, min(len(train_list), (iter_ + 1) * bs)
        data_batch = data[left: right]

        temp_out = np.array([item[0] for item in data_batch])

        a = [item[1] for item in data_batch]
        s = a[0].shape[0]

        b = np.concatenate(a).reshape(-1, s)

        temp_in = torch.from_numpy(b).float()
        temp_target = torch.from_numpy(temp_out).long()

        temp_in, temp_target = cuda_(temp_in), cuda_(temp_target)

        pred = model(temp_in)
        y_true += temp_out.tolist()
        pred_result = pred.data.max(1)[1]
        correct += sum(np.equal(pred_result.cpu().numpy(), temp_out))

        y_pred += pred_result.cpu().numpy().tolist()

    print('Validating purpose {} takes {} seconds, cumulative loss is: {}, accuracy: {}%'.format(purpose, time.time() - start, epoch_loss / max_iter, correct * 100.0 / (max_iter * bs)))
    print(classification_report(y_true, y_pred))
    model.train()


def train(bs, train_list, valid_list, test_list, optimizer, model, criterion, epoch, model_path):
    print('-------validating before training {} epoch-------'.format(epoch))
    if epoch > 0:
        validate(2, train_list, valid_list, test_list, model)

    if epoch == 9:
        PATH = model_path
        torch.save(model.state_dict(), PATH)
        print('Model saved at {}'.format(PATH))
        return

    model.train()
    random.shuffle(train_list)
    epoch_loss = 0
    max_iter = int(len(train_list) / bs)
    start = time.time()
    for iter_ in range(max_iter):
        left, right = iter_ * bs, min(len(train_list), (iter_ + 1) * bs)
        data_batch = train_list[left: right]

        temp_out = np.array([item[0] for item in data_batch])

        a = [item[1] for item in data_batch]
        s = a[0].shape[0]

        b = np.concatenate(a).reshape(-1, s)

        temp_in = torch.from_numpy(b).float()

        temp_target = torch.from_numpy(temp_out).long()

        temp_in, temp_target = cuda_(temp_in), cuda_(temp_target)

        pred = model(temp_in)
        loss = criterion(pred, temp_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.data

        if iter_ % 500 == 0:
            print('{} seconds to finished {}% cumulative loss is: {}'.format(time.time() - start, iter_ * 100.0 / max_iter, epoch_loss / iter_))


def main():
    parser = argparse.ArgumentParser(description="Pretrain Policy Network")
    parser.add_argument('-inputdim', type=int, dest='inputdim', help='input dimension', default = 29)
    parser.add_argument('-hiddendim', type=int, dest='hiddendim', help='hidden dimension', default = 64)
    parser.add_argument('-outputdim', type=int, dest='outputdim', help='output dimension', default = 12)
    parser.add_argument('-bs', type=int, dest='bs', help='batch size', default = 16)
    parser.add_argument('-optim', type=str, dest='optim', help='optimizer choice', default = 'Adam')
    parser.add_argument('-lr', type=float, dest='lr', help='learning rate', default = 0.001)
    parser.add_argument('-decay', type=float, dest='decay', help='weight decay', default = 0)
    parser.add_argument('-mod', type=str, dest='mod', help='mod', default = 'ours') # ear crm

    A = parser.parse_args()
    print('Arguments loaded!')

    PN = PolicyNetwork(input_dim=A.inputdim, dim1=A.hiddendim, output_dim=A.outputdim)

    cuda_(PN)
    print('Model on GPU')
    data_list = list()

    dir = '../data/pretrain-numpy-data-{}'.format(A.mod)

    files = os.listdir(dir)
    file_paths = [dir + '/' + f for f in files]

    i = 0
    for fp in file_paths:
        with open(fp, 'rb') as f:
            try:
                data_list += pickle.load(f)
                i += 1
            except:
                pass
    print('total files: {}'.format(i))
    data_list = data_list[: int(len(data_list))]
    print('length of data list is: {}'.format(len(data_list)))

    random.shuffle(data_list)

    train_list = data_list[: int(len(data_list) * 0.7)]
    valid_list = data_list[int(len(data_list) * 0.7): int(len(data_list) * 0.9)]
    test_list = data_list[int(len(data_list) * 0.9):]
    print('train length: {}, valid length: {}, test length: {}'.format(len(train_list), len(valid_list), len(test_list)))

    if A.optim == 'Ada':
        optimizer = torch.optim.Adagrad(PN.parameters(), lr=A.lr, weight_decay=A.decay)
    if A.optim == 'Adam':
        optimizer = torch.optim.Adam(PN.parameters(), lr=A.lr, weight_decay=A.decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        random.shuffle(train_list)
        model_name = '../data/PN-model-{}/pretrain-model.pt'.format(A.mod)
        train(A.bs, train_list, valid_list, test_list, optimizer, PN, criterion, epoch, model_name)


if __name__ == '__main__':
    main()