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
            S_lst = [x if (x.shape[0]==x.shape[1]) else np.corrcoef(x.T) for x in S_lst]
        else:
            S_lst = [S_.T] if (S_.shape[0]==S_.shape[1]) else [np.corrcoef(S_)]
        if (all([self.sep_feature in str(f) for f in dataset.user_features])):
            types_feature = [str(f).split(self.sep_feature)[0] for f in dataset.user_features]
            P_lst = [P_[:,np.argwhere(np.array(types_feature)==tf)].T for tf in list(set(types_feature))]
            P_lst = [x.reshape(x.shape[1:]) for x in P_lst]
            P_lst = [x if (x.shape[0]==x.shape[1]) else np.corrcoef(x.T) for x in P_lst]
        else:
            P_lst = [P_.T] if (P_.shape[0]==P_.shape[1]) else [np.corrcoef(P_)]
        Y = dataset.ratings.toarray().copy()
        Y[Y==-1] = 0
        keep_ids_dr = (np.sum(Y,axis=1)!=0)
        keep_ids_di = (np.sum(Y,axis=0)!=0)
        P_lst = [P[keep_ids_di,:][:,keep_ids_di] for P in P_lst]
        P_lst = [P for P in P_lst if (not np.isnan(np.mean(P)))]
        S_lst = [S[keep_ids_dr,:][:,keep_ids_dr] for S in S_lst]
        S_lst = [S for S in S_lst if (not np.isnan(np.mean(S)))]
        Y_ = Y[keep_ids_dr,:][:,keep_ids_di]
        return [S_lst, P_lst, Y_, keep_ids_dr, keep_ids_di] if (is_training) else [Y]
        
    ## https://raw.githubusercontent.com/GCQ2119216031/DDA-SKF/dcad0b455f2d436bafe03b03ce07394f54f075e4/src/Novel_drug_prediction.m
    def model_fit(self, S_lst, P_lst, Y, keep_ids_dr, keep_ids_di):
        time_stamp = calendar.timegm(current_GMT)+np.random.choice(range(int(1e8)), size=1)[0]
        filefolder = "DDA_SKF_%s" % time_stamp 
        call("mkdir -p %s/" % filefolder, shell=True)
        call("wget -qO %s/DDA_SKF.m 'https://raw.githubusercontent.com/GCQ2119216031/DDA-SKF/dcad0b455f2d436bafe03b03ce07394f54f075e4/src/Novel_drug_prediction.m'" % filefolder, shell=True)
        call("sed -i '1,56d' %s/DDA_SKF.m" % filefolder, shell=True)
        np.savetxt("%s/Y.csv" % filefolder, Y, delimiter=",")
        for i, S in enumerate(S_lst): 
            np.savetxt("%s/S_%d.csv" % (filefolder, i+1), S, delimiter=",")
        for i, P in enumerate(P_lst): 
            np.savetxt("%s/P_%d.csv" % (filefolder, i+1), P, delimiter=",")
        cmd = ["pkg load statistics; interaction_matrix = csvread('Y.csv')'; [dn,dr] = size(interaction_matrix); train_interaction_matrix = interaction_matrix"]
        cmd1 = ["K1 = []"]+["K1(:,:,%d)=csvread('P_%d.csv')" % (i+1, i+1) for i, _ in enumerate(P_lst)]+["K1(:,:,%d)=interaction_similarity(train_interaction_matrix, '1')" % (len(P_lst)+1)]
        cmd2 = ["K2 = []"]+["K2(:,:,%d)=csvread('S_%d.csv')" % (i+1, i+1) for i, _ in enumerate(S_lst)]+["K2(:,:,%d)=interaction_similarity(train_interaction_matrix, '2')" % (len(S_lst)+1)]
        cmd += cmd1+cmd2
        cmd += ["K_COM1=SKF({"+",".join(["K1(:,:,%d)" % (i+1) for i,_ in enumerate(cmd1[1:])])+"},%d,10,0.2)" % min(12, P.shape[0]-1)]
        cmd += ["K_COM2=SKF({"+",".join(["K2(:,:,%d)" % (i+1) for i,_ in enumerate(cmd1[1:])])+"},%d,6,0.4)" % min(20, S.shape[0]-1)] 
        cmd += ["score_matrix = LapRLS(K_COM1,K_COM2,train_interaction_matrix,%f,%f)" % (self.lamuda, self.beta)]
        cmd += ["csvwrite('score_matrix.csv', real(score_matrix))"]
        cmd = (";\n".join(cmd))+";"
        call("echo \"function DDA_SKF\n%s\nend\n\n\" | cat - %s/DDA_SKF.m > %s/DDA_SKF2.m" % (cmd,filefolder,filefolder), shell=True)
        call("mv %s/DDA_SKF2.m %s/DDA_SKF.m" % (filefolder,filefolder), shell=True)
        call("sed -i 's/squaredeuclidean/sqeuclidean/g' %s/DDA_SKF.m" % filefolder, shell=True)
        call("cd %s/ && octave 'DDA_SKF.m'" % (filefolder), shell=True)
        preds_ = np.loadtxt("%s/score_matrix.csv" % filefolder, delimiter=",").T
        preds = np.zeros((len(keep_ids_dr), len(keep_ids_di)))
        for ii, i in enumerate(np.argwhere(keep_ids_dr)):
            preds[i,keep_ids_di] = preds_[ii,:]
        self.estimator = {
            "predictions" : preds,
        }
        call("rm -rf %s/" % filefolder, shell=True)

    def model_predict_proba(self, Y):
        assert Y.shape==self.estimator["predictions"].shape
        return self.estimator["predictions"]