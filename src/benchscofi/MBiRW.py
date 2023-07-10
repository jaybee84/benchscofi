#coding: utf-8

## https://github.com/bioinfomaticsCSU/MBiRW/tree/d0487b2a43e37a7ee4026959cb052e2527611fde
from stanscofi.models import BasicModel, create_overscores
from stanscofi.preprocessing import CustomScaler

import numpy as np
import pandas as pd
import os
from subprocess import call
import scipy.io
import warnings

import calendar
import time
current_GMT = time.gmtime()

## /!\ Only tested on Linux
class MBiRW(BasicModel):
    def __init__(self, params=None):
        try:
            call("octave -v", shell=True)
        except:
            raise ValueError("Please install Octave.")
        params = params if (params is not None) else self.default_parameters()
        super(MBiRW, self).__init__(params)
        self.scalerS, self.scalerP = None, None
        self.name = "MBiRW" 
        self.model = None
        self.use_masked_dataset = True
        self.MBiRW_filepath = None
        self.alpha = params["alpha"]
        self.l = params["l"]
        self.r = params["r"]
        self.d = params["d"]

    def default_parameters(self):
        params = {
            "decision_threshold": 1, 
            "alpha": 0.3,
            "l":2, "r":2, "d": np.log(9999),
        }
        return params

    def preprocessing(self, dataset, inf=2):
        if (self.scalerS is None):
            self.scalerS = CustomScaler(posinf=inf, neginf=-inf)
        S_ = self.scalerS.fit_transform(dataset.items.T.copy(), subset=None)
        X_s = S_ if (S_.shape[0]==S_.shape[1]) else np.corrcoef(S_)
        if (self.scalerP is None):
            self.scalerP = CustomScaler(posinf=inf, neginf=-inf)
        P_ = self.scalerP.fit_transform(dataset.users.T.copy(), subset=None)
        X_p = P_ if (P_.shape[0]==P_.shape[1]) else np.corrcoef(P_)
        A_sp = dataset.ratings_mat.T # users x items
        return X_s, X_p, A_sp
        
    def fit(self, train_dataset):
        X_s, X_p, A_sp = self.preprocessing(train_dataset)
        time_stamp = calendar.timegm(current_GMT)
        filefolder = "MBiRW_%s" % time_stamp 
        call("mkdir -p %s/" % filefolder, shell=True)
        repo_url = "https://raw.githubusercontent.com/bioinfomaticsCSU/MBiRW/d0487b2a43e37a7ee4026959cb052e2527611fde"
        call("wget -qO %s/normFun.m '%s/Code/normFun.m'" % (filefolder, repo_url), shell=True)
        call("wget -qO %s/setparFun.m '%s/Code/setparFun.m'" % (filefolder, repo_url), shell=True)
        call("wget -qO - '%s/Code/nManiCluester.m' | sed '5,55d' > %s/nManiCluester.m" % (repo_url, filefolder), shell=True)
        call("wget -qO %s/cluster_one-1.0.jar '%s/Code/cluster_one-1.0.jar'" % (filefolder, repo_url), shell=True)
        np.savetxt("%s/X_p.csv" % filefolder, X_p, delimiter=",")
        np.savetxt("%s/A_sp.csv" % filefolder, A_sp, delimiter=",")
        np.savetxt("%s/X_s.csv" % filefolder, X_s, delimiter=",")
        np.savetxt("%s/s_names.csv" % filefolder, train_dataset.item_list, delimiter=",")
        np.savetxt("%s/p_names.csv" % filefolder, train_dataset.user_list, delimiter=",")
        ## https://github.com/bioinfomaticsCSU/MBiRW/blob/d0487b2a43e37a7ee4026959cb052e2527611fde/Code/MBiRW.m
        newWrr, newWdd = np.copy(X_s), np.copy(X_p)
        dr, dn = newWrr.shape[1], newWdd.shape[1]
        with open("%s/DrugsP.txt" % filefolder, "w") as f:
            lines = ["\t".join([str(train_dataset.item_list[i]), str(train_dataset.item_list[j]), str(newWrr[i,j])]) for i in range(dr) for j in range(i) if (newWrr[i,j]>0)]
            f.write("\n".join(lines))
        with open("%s/DiseasesP.txt" % filefolder, "w") as f:
            lines = ["\t".join([str(train_dataset.user_list[i]), str(train_dataset.user_list[j]), str(newWdd[i,j])]) for i in range(dn) for j in range(i) if (newWdd[i,j]>0)]
            f.write("\n".join(lines))
        call('cd %s/ && java -jar "cluster_one-1.0.jar"  "DrugsP.txt" -F csv > DrugsC.txt' % filefolder, shell=True)
        call('cd %s/ && java -jar "cluster_one-1.0.jar"  "DiseasesP.txt" -F csv > DiseasesC.txt' % filefolder, shell=True)
        cmd = "Wdd = csvread('X_p.csv');Wdr = csvread('A_sp.csv');Wrr = csvread('X_s.csv');Wrname = csvread('s_names.csv');Wdname = csvread('p_names.csv');Wrd = Wdr';A = Wrd;alpha=%f;l=%d;r=%d;d=%f;dn = size(Wdd,1);dr = size(Wrr,1);newWrr = csvread('X_s.csv');newWdd = csvread('X_p.csv');cr = setparFun(Wrd,Wrr);cd = setparFun(Wdr,Wdd);LWrr = 1./(1+exp(cr*Wrr+d));LWdd = 1./(1+exp(cd*Wdd+d));[RWrr,RWdd] = nManiCluester(LWrr,LWdd,newWrr,newWdd,Wrname,Wdname);normWrr = normFun(RWrr);normWdd = normFun(RWdd);R0 = A/sum(A(:));Rt = R0;for t=1:max(l,r);ftl = 0;ftr = 0;if(t<=l);nRtleft = alpha * normWrr*Rt + (1-alpha)*R0;ftl = 1;end;if(t<=r);nRtright = alpha * Rt * normWdd + (1-alpha)*R0;ftr = 1;end;Rt =  (ftl*nRtleft + ftr*nRtright)/(ftl + ftr);end;csvwrite('Rt.csv', Rt);" % (self.alpha, self.l, self.r, self.d)
        call("cd %s/ && octave --silent --eval \"%s\"" % (filefolder, cmd), shell=True)
        self.model = np.loadtxt("%s/Rt.csv" % filefolder, delimiter=",")
        call("rm -rf %s/" % filefolder, shell=True)

    def model_predict(self, test_dataset):
        assert test_dataset.folds is not None
        folds = np.copy(test_dataset.folds)
        folds[:,2] = [self.model[user,item] for user, item in folds[:,:2].tolist()]
        scores = create_overscores(folds, test_dataset)
        return scores