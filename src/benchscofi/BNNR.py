#coding: utf-8

## https://github.com/BioinformaticsCSU/BNNR/tree/68f8e98c02459189b6eeac68a86306ccc1da0374
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
class BNNR(BasicModel):
    def __init__(self, params=None):
        try:
            call("octave -v", shell=True)
        except:
            raise ValueError("Please install Octave.")
        params = params if (params is not None) else self.default_parameters()
        super(BNNR, self).__init__(params)
        self.scalerS, self.scalerP = None, None
        self.name = "BNNR" 
        self.estimator = None
        self.BNNR_filepath = None

    def default_parameters(self):
        params = {
            "maxiter": 300,
            "alpha": 1,
            "beta": 10,
            "tol1": 2e-3,
            "tol2": 1e-5,
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
        return [X_s, X_p, A_sp]
        
    def model_fit(self, X_s, X_p, A_sp):
        time_stamp = calendar.timegm(current_GMT)+np.random.choice(range(int(1e8)), size=1)[0]
        filefolder = "BNNR_%s" % time_stamp 
        call("mkdir -p %s/" % filefolder, shell=True)
        call("wget -qO %s/BNNR.m 'https://raw.githubusercontent.com/BioinformaticsCSU/BNNR/master/BNNR.m'" % filefolder, shell=True)
        call("wget -qO %s/svt.m 'https://raw.githubusercontent.com/BioinformaticsCSU/BNNR/master/svt.m'" % filefolder, shell=True)
        np.savetxt("%s/X_p.csv" % filefolder, X_p, delimiter=",")
        np.savetxt("%s/A_sp.csv" % filefolder, A_sp, delimiter=",")
        np.savetxt("%s/X_s.csv" % filefolder, X_s, delimiter=",")
        cmd = "Wdd = csvread('X_p.csv');Wdr = csvread('A_sp.csv');Wrr = csvread('X_s.csv');T = [Wrr, Wdr'; Wdr, Wdd];[WW,iter] = BNNR(%d, %d, T, double(T ~= 0), %f, %f, %d, 0, 1);[t1, t2] = size(T);[dn,dr] = size(Wdr);M_recovery = WW((t1-dn+1) : t1, 1 : dr);csvwrite('M_recovery.csv', M_recovery);csvwrite('iter.csv', iter);" % (self.alpha, self.beta, self.tol1, self.tol2, self.maxiter)
        call("cd %s/ && octave --silent --eval \"%s\"" % (filefolder, cmd), shell=True)
        self.estimator = {
            "niter" : int(np.loadtxt("%s/iter.csv" % filefolder, delimiter=",")),
            "predictions" : np.loadtxt("%s/M_recovery.csv" % filefolder, delimiter=",").T,
        }
        call("rm -rf %s/" % filefolder, shell=True)

    def model_predict_proba(self, X_s, X_p, A_sp):
        return self.estimator["predictions"]