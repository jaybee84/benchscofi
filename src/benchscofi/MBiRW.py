#coding: utf-8

## https://github.com/bioinfomaticsCSU/MBiRW/tree/d0487b2a43e37a7ee4026959cb052e2527611fde
from stanscofi.models import BasicModel
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
        self.predictions = None
        self.MBiRW_filepath = None

    def default_parameters(self):
        params = {
            "alpha": 0.3, "l":2, "r":2, "d": np.log(9999),
        }
        return params

    def preprocessing(self, dataset, is_training=True, inf=2):
        if (self.scalerS is None):
            self.scalerS = CustomScaler(posinf=inf, neginf=-inf)
        S_ = self.scalerS.fit_transform(dataset.items.T.toarray().copy(), subset=None)
        X_s = S_ if (S_.shape[0]==S_.shape[1]) else np.corrcoef(S_)
        if (self.scalerP is None):
            self.scalerP = CustomScaler(posinf=inf, neginf=-inf)
        P_ = self.scalerP.fit_transform(dataset.users.T.toarray().copy(), subset=None)
        X_p = P_ if (P_.shape[0]==P_.shape[1]) else np.corrcoef(P_)
        A_sp = dataset.ratings.toarray().T # users x items
        return [X_s, X_p, A_sp, np.array(dataset.item_list), np.array(dataset.user_list)] if (is_training) else [X_s, X_p, A_sp]
        
    def model_fit(self, X_s, X_p, A_sp, item_list, user_list):
        time_stamp = calendar.timegm(current_GMT)+np.random.choice(range(int(1e8)), size=1)[0]
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
        with open("%s/s_names.csv" % filefolder, "w") as f:
            f.write(",".join(item_list))
        with open("%s/p_names.csv" % filefolder, "w") as f:
            f.write(",".join(user_list))
        ## https://github.com/bioinfomaticsCSU/MBiRW/blob/d0487b2a43e37a7ee4026959cb052e2527611fde/Code/MBiRW.m
        newWrr, newWdd = np.copy(X_s), np.copy(X_p)
        dr, dn = newWrr.shape[1], newWdd.shape[1]
        with open("%s/DrugsP.txt" % filefolder, "w") as f:
            lines = ["\t".join([str(item_list[i]), str(item_list[j]), str(newWrr[i,j])]) for i in range(dr) for j in range(i) if (newWrr[i,j]>0)]
            f.write("\n".join(lines))
        with open("%s/DiseasesP.txt" % filefolder, "w") as f:
            lines = ["\t".join([str(user_list[i]), str(user_list[j]), str(newWdd[i,j])]) for i in range(dn) for j in range(i) if (newWdd[i,j]>0)]
            f.write("\n".join(lines))
        call('cd %s/ && java -jar "cluster_one-1.0.jar"  "DrugsP.txt" -F csv > DrugsC.txt' % filefolder, shell=True)
        call('cd %s/ && java -jar "cluster_one-1.0.jar"  "DiseasesP.txt" -F csv > DiseasesC.txt' % filefolder, shell=True)
        cmd = "Wdd = csvread('X_p.csv');Wdr = csvread('A_sp.csv');Wrr = csvread('X_s.csv');Wrname = csvread('s_names.csv');Wdname = csvread('p_names.csv');Wrd = Wdr';A = Wrd;alpha=%f;l=%d;r=%d;d=%f;dn = size(Wdd,1);dr = size(Wrr,1);newWrr = csvread('X_s.csv');newWdd = csvread('X_p.csv');cr = setparFun(Wrd,Wrr);cd = setparFun(Wdr,Wdd);LWrr = 1./(1+exp(cr*Wrr+d));LWdd = 1./(1+exp(cd*Wdd+d));[RWrr,RWdd] = nManiCluester(LWrr,LWdd,newWrr,newWdd,Wrname,Wdname);normWrr = normFun(RWrr);normWdd = normFun(RWdd);R0 = A/sum(A(:));Rt = R0;for t=1:max(l,r);ftl = 0;ftr = 0;if(t<=l);nRtleft = alpha * normWrr*Rt + (1-alpha)*R0;ftl = 1;end;if(t<=r);nRtright = alpha * Rt * normWdd + (1-alpha)*R0;ftr = 1;end;Rt =  (ftl*nRtleft + ftr*nRtright)/(ftl + ftr);end;csvwrite('Rt.csv', Rt);" % (self.alpha, self.l, self.r, self.d)
        call("cd %s/ && octave --silent --eval \"%s\"" % (filefolder, cmd), shell=True)
        self.predictions = np.loadtxt("%s/Rt.csv" % filefolder, delimiter=",")
        call("rm -rf %s/" % filefolder, shell=True)

    def model_predict_proba(self, X_s, X_p, A_sp):
        return self.predictions