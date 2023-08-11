#coding: utf-8

#https://github.com/ljatynu/NIMCGCN/tree/a0798ed29ae329dd71bff096ffc678527cc4099e/
from stanscofi.models import BasicModel
from stanscofi.preprocessing import CustomScaler

from benchscofi.implementations import NIMCGCNImplementation

import numpy as np
import random
import torch as t
from torch import optim
import os
from subprocess import Popen

def read_mat(df):
    return t.FloatTensor([np.asarray(df[irow,:],dtype=float).tolist() for irow in range(df.shape[0])])

class NIMCGCN(BasicModel):
    def __init__(self, params=None):
        params = params if (params is not None) else self.default_parameters()
        super(NIMCGCN, self).__init__(params)
        self.scalerS, self.scalerP = None, None
        self.validation=1 ## placeholder
        self.name = "NIMCGCN" 
        self.model = None

    def default_parameters(self):
        params = {
            "epoch" : 10,
            "alpha" : 0.2,
            "fg" : 256,
            "fd" : 256,
            "k" : 32,
            "display_epoch": 5,
            "random_state": 1234, 
            "learning_rate": 0.001,
        }
        return params

    def preprocessing(self, dataset, is_training=True, inf=2):
        if (self.scalerS is None):
            self.scalerS = CustomScaler(posinf=inf, neginf=-inf)
        S_ = self.scalerS.fit_transform(dataset.items.T.toarray().copy(), subset=None)
        S_ = np.nan_to_num(S_, nan=0.0) ##
        X_s = S_ if (S_.shape[0]==S_.shape[1]) else np.corrcoef(S_)
        if (self.scalerP is None):
            self.scalerP = CustomScaler(posinf=inf, neginf=-inf)
        P_ = self.scalerP.fit_transform(dataset.users.T.toarray().copy(), subset=None)
        P_ = np.nan_to_num(P_, nan=0.0) ##
        X_p = P_ if (P_.shape[0]==P_.shape[1]) else np.corrcoef(P_)
        A_sp = dataset.ratings.toarray().T # users x items
        ## https://github.com/ljatynu/NIMCGCN/tree/a0798ed29ae329dd71bff096ffc678527cc4099e
        ## do not take into account negative samples
        A_sp[A_sp==-1] = 0
        dataset_ = dict()
        dataset_['md_p'] = read_mat(A_sp)
        dataset_['md_true'] = read_mat(A_sp)

        zero_index = []
        one_index = []
        for i in range(dataset_['md_p'].size(0)):
            for j in range(dataset_['md_p'].size(1)):
                if dataset_['md_p'][i][j] < 1:
                    zero_index.append([i, j])
                if dataset_['md_p'][i][j] >= 1:
                    one_index.append([i, j])
        random.shuffle(one_index)
        random.shuffle(zero_index)
        zero_tensor = t.LongTensor(zero_index)
        one_tensor = t.LongTensor(one_index)
        dataset_['md'] = dict()
        dataset_['md']['train'] = [one_tensor, zero_tensor]

        dd_matrix = read_mat(X_s)
        dd_edge_index = NIMCGCNImplementation.get_edge_index(dd_matrix)
        dataset_['dd'] = {'data': dd_matrix, 'edge_index': dd_edge_index}

        mm_matrix = read_mat(X_p)
        mm_edge_index = NIMCGCNImplementation.get_edge_index(mm_matrix)
        dataset_['mm'] = {'data': mm_matrix, 'edge_index': mm_edge_index}

        sizes = NIMCGCNImplementation.Sizes(dataset_)
        dat = NIMCGCNImplementation.Dataset(self, dataset_)
        return [dat[0], X_s.shape[0], X_p.shape[0]] if (is_training) else [dat[0]]
        
    def model_fit(self, train_dat, d, m):
        t.manual_seed(self.random_state)
        np.random.seed(self.random_state) 
        random.seed(self.random_state)
        self.d, self.m = d, m
        self.model = NIMCGCNImplementation.Model(self)
        self.model.alpha = self.alpha
        self.model.epoch = self.epoch
        if (t.cuda.is_available()):
            model.cuda()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        NIMCGCNImplementation.train(self.model, train_dat, optimizer, self)

    def model_predict_proba(self, test_dat):
        outputs = self.model(test_dat)
        y_pred = outputs.detach().numpy().T
        return y_pred
