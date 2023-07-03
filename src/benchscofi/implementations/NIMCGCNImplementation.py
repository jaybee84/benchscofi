#coding:utf-8

## Concatenated contents of https://github.com/ljatynu/NIMCGCN/tree/a0798ed29ae329dd71bff096ffc678527cc4099e/code
from __future__ import division
from torch import nn, optim


import torch as t
from torch import nn
from torch_geometric.nn import conv

import csv
import torch as t
import random

class Config(object):
    def __init__(self):
        self.data_path = '../data'
        self.validation = 5
        self.save_path = '../data'
        self.epoch = 300
        self.alpha = 0.2


class Myloss(nn.Module):
    def __init__(self, alpha=1.):
        self.alpha = alpha
        super(Myloss, self).__init__()

    def forward(self, one_index, zero_index, target, input):
        loss = nn.MSELoss(reduction='none')
        loss_sum = loss(input, target)
        return (1-self.alpha)*loss_sum[one_index].sum()+self.alpha*loss_sum[zero_index].sum()


class Sizes(object):
    def __init__(self, dataset):
        self.m = dataset['mm']['data'].size(0)
        self.d = dataset['dd']['data'].size(0)
        self.fg = 256
        self.fd = 256
        self.k = 32


def train(model, train_data, optimizer, opt):
    model.train()
    regression_crit = Myloss(model.alpha)
    one_index = train_data[2][0].cuda().t().tolist() if (t.cuda.is_available()) else train_data[2][0].t().tolist()
    zero_index = train_data[2][1].cuda().t().tolist() if (t.cuda.is_available()) else train_data[2][1].t().tolist()

    def train_epoch():
        model.zero_grad()
        score = model(train_data)
        loss = regression_crit(one_index, zero_index, train_data[4].cuda() if (t.cuda.is_available()) else train_data[4], score)
        loss.backward()
        optimizer.step()
        return loss
    for epoch in range(1, model.epoch+1):
        train_reg_loss = train_epoch()
        if (epoch%opt.display_epoch==0):
            print("LOSS %.5f (epoch %d)" % (train_reg_loss.item()/(len(one_index[0])+len(zero_index[0])), epoch))



def main():
    opt = Config()
    dataset = prepare_data(opt)
    sizes = Sizes(dataset)
    train_data = Dataset(opt, dataset)
    for i in range(opt.validation):
        print('-'*50)
        model = Model(sizes)
        #model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train(model, train_data[i], optimizer, opt)



def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return t.FloatTensor(md_data)


def read_txt(path):
    with open(path, 'r', newline='') as txt_file:
        reader = txt_file.readlines()
        md_data = []
        md_data += [[float(i) for i in row.split()] for row in reader]
        return t.FloatTensor(md_data)


def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return t.LongTensor(edge_index)


def prepare_data(opt):
    dataset = dict()
    dataset['md_p'] = read_csv(opt.data_path + '\\m-d.csv')
    dataset['md_true'] = read_csv(opt.data_path + '\\m-d.csv')

    zero_index = []
    one_index = []
    for i in range(dataset['md_p'].size(0)):
        for j in range(dataset['md_p'].size(1)):
            if dataset['md_p'][i][j] < 1:
                zero_index.append([i, j])
            if dataset['md_p'][i][j] >= 1:
                one_index.append([i, j])
    random.shuffle(one_index)
    random.shuffle(zero_index)
    zero_tensor = t.LongTensor(zero_index)
    one_tensor = t.LongTensor(one_index)
    dataset['md'] = dict()
    dataset['md']['train'] = [one_tensor, zero_tensor]

    dd_matrix = read_csv(opt.data_path + '\\d-d.csv')
    dd_edge_index = get_edge_index(dd_matrix)
    dataset['dd'] = {'data': dd_matrix, 'edge_index': dd_edge_index}

    mm_matrix = read_csv(opt.data_path + '\\m-m.csv')
    mm_edge_index = get_edge_index(mm_matrix)
    dataset['mm'] = {'data': mm_matrix, 'edge_index': mm_edge_index}
    return dataset



class Dataset(object):
    def __init__(self, opt, dataset):
        self.data_set = dataset
        self.nums = opt.validation

    def __getitem__(self, index):
        return (self.data_set['dd'], self.data_set['mm'],
                self.data_set['md']['train'], None,
                self.data_set['md_p'], self.data_set['md_true'])

    def __len__(self):
        return self.nums




class Model(nn.Module):
    def __init__(self, sizes):
        super(Model, self).__init__()

        self.fg = sizes.fg
        self.fd = sizes.fd
        self.k = sizes.k
        self.m = sizes.m
        self.d = sizes.d
        self.gcn_x1 = conv.GCNConv(self.fg, self.fg)
        self.gcn_y1 = conv.GCNConv(self.fd, self.fd)
        self.gcn_x2 = conv.GCNConv(self.fg, self.fg)
        self.gcn_y2 = conv.GCNConv(self.fd, self.fd)

        self.linear_x_1 = nn.Linear(self.fg, 256)
        self.linear_x_2 = nn.Linear(256, 128)
        self.linear_x_3 = nn.Linear(128, 64)

        self.linear_y_1 = nn.Linear(self.fd, 256)
        self.linear_y_2 = nn.Linear(256, 128)
        self.linear_y_3 = nn.Linear(128, 64)

    def forward(self, input):
        #t.manual_seed(1)
        x_m = t.randn(self.m, self.fg)
        x_d = t.randn(self.d, self.fd)

        ## ensure that weights are nonnegative
        weights1 = t.relu(input[1]['data'][input[1]['edge_index'][0], input[1]['edge_index'][1]].cuda() if (t.cuda.is_available()) else input[1]['data'][input[1]['edge_index'][0], input[1]['edge_index'][1]])
        X1 = t.relu(self.gcn_x1(x_m.cuda() if (t.cuda.is_available()) else x_m
		, input[1]['edge_index'].cuda() if (t.cuda.is_available()) else input[1]['edge_index']
		, weights1
        ))

        ## ensure that weights are nonnegative
        weights2 = t.relu(input[1]['data'][input[1]['edge_index'][0], input[1]['edge_index'][1]].cuda() if (t.cuda.is_available()) else input[1]['data'][input[1]['edge_index'][0], input[1]['edge_index'][1]])
        X = t.relu(self.gcn_x2(X1, input[1]['edge_index'].cuda() if (t.cuda.is_available()) else input[1]['edge_index']
            , weights2
        ))

        ## ensure that weights are nonnegative
        weights3 = t.relu(input[0]['data'][input[0]['edge_index'][0], input[0]['edge_index'][1]].cuda() if (t.cuda.is_available()) else input[0]['data'][input[0]['edge_index'][0], input[0]['edge_index'][1]])
        Y1 = t.relu(self.gcn_y1(x_d.cuda() if (t.cuda.is_available()) else x_d, input[0]['edge_index'].cuda() if (t.cuda.is_available()) else input[0]['edge_index']
            , weights3
        ))
        ## ensure that weights are nonnegative
        weights4 = t.relu(input[0]['data'][input[0]['edge_index'][0], input[0]['edge_index'][1]].cuda() if (t.cuda.is_available()) else input[0]['data'][input[0]['edge_index'][0], input[0]['edge_index'][1]] )
        Y = t.relu(self.gcn_y2(Y1, input[0]['edge_index'].cuda() if (t.cuda.is_available()) else input[0]['edge_index']
            , weights4
        ))

        x1 = t.relu(self.linear_x_1(X))
        x2 = t.relu(self.linear_x_2(x1))
        x = t.relu(self.linear_x_3(x2))

        y1 = t.relu(self.linear_y_1(Y))
        y2 = t.relu(self.linear_y_2(y1))
        y = t.relu(self.linear_y_3(y2))

        return x.mm(y.t())


