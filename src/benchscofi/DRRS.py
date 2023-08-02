#coding: utf-8

## http://bioinformatics.csu.edu.cn/resources/softs/DrugRepositioning/DRRS/index.html
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
class DRRS(BasicModel):
    def __init__(self, params=None):
        self.MCR_HOME="/usr/local/MATLAB/MATLAB_Compiler_Runtime"
        self.DRRS_path = "http://bioinformatics.csu.edu.cn/resources/softs/DrugRepositioning/DRRS/soft/"
        if (not os.path.exists(self.MCR_HOME)):
            raise ValueError("Please install MATLAB.")
        params = params if (params is not None) else self.default_parameters()
        super(DRRS, self).__init__(params)
        assert self.use_linux
        self.scalerS, self.scalerP = None, None
        self.name = "DRRS" 
        self.predictions = None
        self.DRRS_filepath = "DRRS_L" if (self.use_linux) else "DRRS_W.exe"

    def default_parameters(self):
        params = {
            "use_linux": True, #False: use windows
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
        return [X_s, X_p, A_sp]
        
    def model_fit(self, X_s, X_p, A_sp):
        time_stamp = calendar.timegm(current_GMT)
        filefolder = "DRRS_%s" % time_stamp 
        call("mkdir -p %s/" % filefolder, shell=True) 
        call("wget -qO "+filefolder+"/"+self.DRRS_filepath+" "+self.DRRS_path+self.DRRS_filepath+" && chmod +x "+filefolder+"/"+self.DRRS_filepath, shell=True)      
        drugsim, diseasesim, didra = [x+".txt" for x in ["DrugSim","DiseaseSim","DiDrA"]]
        for x, tx in zip([X_s,X_p,A_sp],[drugsim, diseasesim, didra]):
            pd.DataFrame(x, index=range(x.shape[0]), columns=range(x.shape[1])).to_csv(filefolder+"/"+tx,sep="\t",header=None,index=None)
        os.environ['LD_LIBRARY_PATH'] = "%s/v80/runtime/glnxa64:%s/v80/bin/glnxa64:%s/v80/sys/java/jre/glnxa64/jre/lib/amd64/server:%s/v80/sys/os/glnxa64:%s/v80/sys/java/jre/glnxa64/jre/lib/amd64:%s/v80/sys/java/jre/glnxa64/jre/lib/amd64/native_threads" % tuple([self.MCR_HOME]*6)
        os.environ['XAPPLRESDIR'] = "%s/v80/X11/app-defaults" % self.MCR_HOME
        call(" ".join(["cd", "%s/" % filefolder, "&&", "./"+self.DRRS_filepath, drugsim, diseasesim, didra]), shell=True)
        assert os.path.exists("%s/Result_dr_Mat.txt" % filefolder)
        self.predictions = np.loadtxt("%s/Result_dr_Mat.txt" % filefolder, delimiter="\t").T
        call("rm -rf %s/ %s" % (filefolder, self.DRRS_filepath), shell=True)

    def model_predict_proba(self, X_s, X_p, A_sp):
        return self.predictions