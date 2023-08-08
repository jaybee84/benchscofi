#coding: utf-8

# https://github.com/GCQ2119216031/DDA-SKF/tree/dcad0b455f2d436bafe03b03ce07394f54f075e4
from stanscofi.models import BasicModel
from stanscofi.preprocessing import CustomScaler

import numpy as np
import pandas as pd
import os
from subprocess import call

import calendar
import time
current_GMT = time.gmtime()

## /!\ Only tested on Linux
class DDA_SKF(BasicModel):
    def __init__(self, params=None):
        try:
            call("octave -v", shell=True)
        except:
            raise ValueError("Please install Octave.")
        try:
            call("octave --silent --eval \"pkg load statistics;\"", shell=True)
        except:
            raise ValueError("Please install Octave package statistics.")
        params = params if (params is not None) else self.default_parameters()
        super(DDA_SKF, self).__init__(params)
        self.scalerS, self.scalerP = None, None
        self.name = "DDA_SKF" 
        self.estimator = None
        self.DDA_SKF_filepath = None

    def default_parameters(self):
        params = {
            "beta" : 0.4, "lamuda" : 2**(-16), 
            "sep_feature": "-",
        }
        return params

    def preprocessing(self, dataset, is_training=True, inf=2):
        if (self.scalerS is None):
            self.scalerS = CustomScaler(posinf=inf, neginf=-inf)
        S_ = self.scalerS.fit_transform(dataset.items.T.toarray().copy(), subset=None)
        if (self.scalerP is None):
            self.scalerP = CustomScaler(posinf=inf, neginf=-inf)
        P_ = self.scalerP.fit_transform(dataset.users.T.toarray().copy(), subset=None)
        if (all([self.sep_feature in str(f) for f in dataset.item_features])):
            types_feature = [str(f).split(self.sep_feature)[0] for f in dataset.item_features]
            S_lst = [S_[:,np.argwhere(np.array(types_feature)==tf)].T for tf in list(set(types_feature))]
            S_lst = [x.reshape(x.shape[1:]) for x in S_lst]
        else:
            S_lst = [S_.T] if (S_.shape[0]==S_.shape[1]) else [np.corrcoef(S_)]
        if (all([self.sep_feature in str(f) for f in dataset.user_features])):
            types_feature = [str(f).split(self.sep_feature)[0] for f in dataset.user_features]
            P_lst = [P_[:,np.argwhere(np.array(types_feature)==tf)].T for tf in list(set(types_feature))]
            P_lst = [x.reshape(x.shape[1:]) for x in P_lst]
        else:
            P_lst = [P_.T] if (P_.shape[0]==P_.shape[1]) else [np.corrcoef(P_)]
        Y = dataset.ratings.toarray().copy()
        Y[Y==-1] = 0
        keep_ids_dr = (np.sum(Y,axis=1)!=0)
        keep_ids_di = (np.sum(Y,axis=0)!=0)
        #print(Y.shape)
        #print()
        P_lst = [P[keep_ids_di,keep_ids_di] for P in P_lst]
        S_lst = [S[keep_ids_dr,keep_ids_dr] for S in S_lst]
        Y = Y[keep_ids_dr,keep_ids_di]
        #print(Y.shape)
        return [S_lst, P_lst, Y] if (is_training) else [S_lst, P_lst, keep_ids_dr, keep_ids_di]
        
    ## https://raw.githubusercontent.com/GCQ2119216031/DDA-SKF/dcad0b455f2d436bafe03b03ce07394f54f075e4/src/Novel_drug_prediction.m
    def model_fit(self, S_lst, P_lst, Y):
        time_stamp = calendar.timegm(current_GMT)
        filefolder = "DDA_SKF_%s" % time_stamp 
        call("mkdir -p %s/" % filefolder, shell=True)
        call("wget -qO %s/DDA_SKF.m 'https://raw.githubusercontent.com/GCQ2119216031/DDA-SKF/dcad0b455f2d436bafe03b03ce07394f54f075e4/src/Novel_drug_prediction.m'" % filefolder, shell=True)
        call("sed -i '1,56d' %s/DDA_SKF.m" % filefolder, shell=True)
        np.savetxt("%s/Y.csv" % filefolder, Y, delimiter=",")
        for i, S in enumerate(S_lst): 
            np.savetxt("%s/S_%d.csv" % (filefolder, i+1), S, delimiter=",")
        for i, P in enumerate(P_lst): 
            np.savetxt("%s/P_%d.csv" % (filefolder, i+1), P, delimiter=",")
        cmd = ["pkg load statistics; interaction_matrix = csvread('Y.csv')'; [dn,dr] = size(interaction_matrix); [~,col] = size(interaction_matrix); colIndex = (1 : col)'; final = zeros(dn,dr); train_interaction_matrix = interaction_matrix"]
        cmd1 = ["K1 = []"]+["K1(:,:,%d)=csvread('P_%d.csv')" % (i+1, i+1) for i, _ in enumerate(P_lst)]+["K1(:,:,%d)=interaction_similarity(train_interaction_matrix, '1')" % (len(P_lst)+1)]
        cmd2 = ["K2 = []"]+["K2(:,:,%d)=csvread('S_%d.csv')" % (i+1, i+1) for i, _ in enumerate(S_lst)]+["K2(:,:,%d)=interaction_similarity(train_interaction_matrix, '2')" % (len(S_lst)+1)]
        cmd += cmd1+cmd2
        cmd += ["K_COM1=SKF({"+",".join(["K1(:,:,%d)" % (i+1) for i,_ in enumerate(cmd1[1:])])+"},%d,10,0.2)" % min(12, P.shape[0]-1)]
        cmd += ["K_COM2=SKF({"+",".join(["K2(:,:,%d)" % (i+1) for i,_ in enumerate(cmd1[1:])])+"},%d,6,0.4)" % min(20, P.shape[0]-1)] 
        cmd += ["score_matrix = LapRLS(K_COM1,K_COM2,train_interaction_matrix,%f,%f)" % (self.lamuda, self.beta)]
        #cmd += ["W1 = K_COM1; W2 = K_COM2; inter3=train_interaction_matrix; lambda = %f; beta = %f" % (self.lamuda, self.beta)]
        #cmd += ["[num_1,num_2] = size(inter3);S_1 = W1;d_1 = sum(S_1);D_1 = diag(d_1);L_D_1 = D_1 - S_1;d_tmep_1=eye(num_1)/(D_1^(1/2));L_D_11 = d_tmep_1*L_D_1*d_tmep_1;A_1 = W1*pinv(W1 + lambda*L_D_11*W1)*inter3"]
        #cmd += ["S_2 = W2;d_2 = sum(S_2);D_2 = diag(d_2);L_D_2 = D_2 - S_2;d_tmep_2=eye(num_2)/(D_2^(1/2));L_D_22 = d_tmep_2*L_D_2*d_tmep_2;A_2 = W2*pinv(W2 + lambda*L_D_22*W2)*inter3'"]
        #cmd += ["LapA=beta*A_1+(1-beta)*A_2'"]
        #cmd += ["score_matrix = LapA"]
        cmd += ["csvwrite('score_matrix.csv', real(score_matrix))"]
        cmd = (";\n".join(cmd))+";"
        call("echo \"function DDA_SKF\n%s\nend\n\n\" | cat - %s/DDA_SKF.m > %s/DDA_SKF2.m" % (cmd,filefolder,filefolder), shell=True)
        call("mv %s/DDA_SKF2.m %s/DDA_SKF.m" % (filefolder,filefolder), shell=True)
        call("sed -i 's/squaredeuclidean/sqeuclidean/g' %s/DDA_SKF.m" % filefolder, shell=True)
        call("cd %s/ && octave --silent --eval \"source('DDA_SKF.m');\"" % (filefolder), shell=True)
        self.estimator = {
            "predictions" : np.loadtxt("%s/score_matrix.csv" % filefolder, delimiter=",").T,
        }
        call("rm -rf %s/" % filefolder, shell=True)

    def model_predict_proba(self, S_lst, P_lst, keep_ids_dr, keep_ids_di):
        preds = np.zeros((len(keep_ids_dr), len(keep_ids_di)))
        preds[keep_ids_dr,keep_ids_di] = self.estimator["predictions"]
        return preds