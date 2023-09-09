#coding: utf-8

## https://github.com/luckymengmeng/SCPMF/tree/9158aa22d53687938bf6402dbcd9812aabf557e9
from stanscofi.models import BasicModel
from stanscofi.preprocessing import CustomScaler

import numpy as np
import os
from subprocess import call

import calendar
import time
current_GMT = time.gmtime()

## /!\ Only tested on Linux
class SCPMF(BasicModel):
    def __init__(self, params=None):
        try:
            call("octave -v", shell=True)
        except:
            raise ValueError("Please install Octave.")
        params = params if (params is not None) else self.default_parameters()
        super(SCPMF, self).__init__(params)
        self.scalerS, self.scalerP = None, None
        self.name = "SCPMF" 
        self.predictions = None
        self.SCPMF_filepath = None

    def default_parameters(self):
        params = {
            "r": 15,
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
        A_sp = np.copy(dataset.ratings.toarray()) # items x users
        return [X_s, X_p, A_sp]
        
    def model_fit(self, X_s, X_p, A_sp):
        time_stamp = calendar.timegm(current_GMT)+np.random.choice(range(int(1e8)), size=1)[0]
        filefolder = "SCPMF_%s" % time_stamp 
        call("mkdir -p %s/" % filefolder, shell=True)
        call("wget -qO %s/SCPMFDR.m 'https://raw.githubusercontent.com/luckymengmeng/SCPMF/master/SCPMFDR.m'" % filefolder, shell=True)
        np.savetxt("%s/X_p.csv" % filefolder, X_p, delimiter=",")
        np.savetxt("%s/A_sp.csv" % filefolder, A_sp, delimiter=",")
        np.savetxt("%s/X_s.csv" % filefolder, X_s, delimiter=",")
        cmd = "A_sp = csvread('A_sp.csv');X_s = csvread('X_s.csv');X_p = csvread('X_p.csv');recMatrix=SCPMFDR(A_sp,X_s,X_p,%d);csvwrite('recMatrix.csv', recMatrix);" % (self.r)
        call("cd %s/ && octave --silent --eval \"%s\"" % (filefolder, cmd), shell=True)
        self.predictions = np.loadtxt("%s/recMatrix.csv" % filefolder, delimiter=",")
        call("rm -rf %s/" % filefolder, shell=True)
  
    def model_predict_proba(self, X_s, X_p, A_sp):
        return self.predictions