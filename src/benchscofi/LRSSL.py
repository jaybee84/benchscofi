#coding: utf-8

from stanscofi.models import BasicModel, create_scores
from stanscofi.preprocessing import CustomScaler

import numpy as np
from subprocess import call
from functools import reduce

import calendar
import time
current_GMT = time.gmtime()

class LRSSL(BasicModel):
    def __init__(self, params=None):
        params = params if (params is not None) else self.default_parameters()
        super(LRSSL, self).__init__(params)
        self.scalerS, self.scalerP = None, None
        self.sep_feature = params.get("sep_feature", "-")
        self.name = "LRSSL"
        self.model = None
        self.k, self.mu, self.lam, self.gam, self.maxiter, self.tol = [params[p] for p in ["k","mu","lam","gam","maxiter","tol"]]
        self.use_masked_dataset = True

    def default_parameters(self):
        params = {
            "decision_threshold": 2e-4, 
            "k": 10, "mu": 0.01, "lam": 0.01, "gam": 2, "tol": 1e-2, "maxiter": 8000, 
            "sep_feature": "-", 
        }
        return params

    def preprocessing(self, dataset, inf=2):
        if (self.scalerS is None):
            self.scalerS = CustomScaler(posinf=inf, neginf=-inf)
        S_ = self.scalerS.fit_transform(dataset.items.T.copy(), subset=None)
        if (self.scalerP is None):
            self.scalerP = CustomScaler(posinf=inf, neginf=-inf)
        P_ = self.scalerP.fit_transform(dataset.users.T.copy(), subset=None)
        if (all([self.sep_feature in str(f) for f in dataset.item_features])):
            types_feature = [str(f).split(self.sep_feature)[0] for f in dataset.item_features]
            X_lst = [S_[:,np.argwhere(np.array(types_feature)==tf)].T for tf in list(set(types_feature))]
        else:
            X_lst = [S_.T]
        X_p = P_ if (P_.shape[0]==P_.shape[1]) else np.corrcoef(P_)
        S_lst = [x.T.dot(x)/np.sqrt(x.sum(axis=0).dot(x.sum(axis=0).T)) for x in X_lst]
        S_lst = [s-np.diag(np.diag(s)) for s in S_lst]
        Y = dataset.ratings_mat.copy()
        ST = np.zeros((Y.shape[0], Y.shape[0]))
        for i in range(Y.shape[0]):
            for j in range(i+1,Y.shape[0]):
                s = X_p[Y[i,:]==1,:][:,Y[j,:]==1]
                if (s.shape[1]==0 or s.shape[0]==0):
                    continue
                ST[i,j] = np.max(s)
        ST = ST + ST.T
        S_lst.append(ST)
        X_lst = [x.T if (x.shape[0]==x.shape[1]) else np.corrcoef(x.T) for x in X_lst]
        return X_lst, S_lst, Y
        
    def fit(self, train_dataset):
        X_lst, S_lst, Y = self.preprocessing(train_dataset, inf=2)
        time_stamp = calendar.timegm(current_GMT)
        cmd = "mkdir -p %s/ && wget -qO - \'https://raw.githubusercontent.com/LiangXujun/LRSSL/a16a75c028393e7256e3630bc8b7900026061f99/LRSSL.R\' | sed -n \'/###/q;p\' > %s/LRSSL.R" % (time_stamp, time_stamp)
        call(cmd, shell=True)
        L_lst = []
        for s in S_lst:
            np.savetxt("%s/s.csv" % time_stamp,s)
            call("R -q -e 'source(\"%s/LRSSL.R\");S <- as.matrix(read.table(\"%s/s.csv\", sep=\" \", header=F));Sp <- get.knn.graph(S, %d);write.csv(S, \"%s/s.csv\", row.names=F)' 2>&1 >/dev/null" % (time_stamp, time_stamp, self.k, time_stamp), shell=True)
            L_lst.append(np.loadtxt("%s/s.csv" % time_stamp, skiprows=1, delimiter=","))
        L_lst = [np.diag(L.sum(axis=0))-L for L in L_lst]
        for i, x in enumerate(X_lst):
            np.savetxt("%s/x_%d.csv" % (time_stamp, i+1),x)
        for i, l in enumerate(L_lst):
            np.savetxt("%s/l_%d.csv" % (time_stamp, i+1),l)
        np.savetxt("%s/y.csv" % time_stamp, Y)
        assert all([s.shape[0]==Y.shape[0] and s.shape[1]==Y.shape[0] for s in S_lst])
        assert all([l.shape[0]==Y.shape[0] and l.shape[1]==Y.shape[0] for l in L_lst])
        assert all([x.shape[0]==Y.shape[0] and x.shape[1]==Y.shape[0] for x in X_lst])
        call("R -q -e 'source(\"%s/LRSSL.R\");ml <- %d;mx <- %d;X_lst <- lapply(1:mx, function(i) as.matrix(read.table(paste0(\"%s/x_\",i,\".csv\"), sep=\" \", header=F)));L_lst <- lapply(1:ml, function(i) as.matrix(read.table(paste0(\"%s/l_\",i,\".csv\"), sep=\" \", header=F)));Y <- as.matrix(read.table(\"%s/y.csv\", sep=\" \", header=F));train.res <- lrssl(X_lst, L_lst, Y, mx, ml, %f, %f, %f, %d, %f);for(i in 1:mx){write.csv(train.res$Gs[[i]], paste0(\"%s/G_\",i,\".csv\"), row.names=F)};write.csv(train.res$alpha, \"%s/alpha.csv\", row.names=F);write.csv(list(t=train.res$t,diff_G=train.res$diff.G), \"%s/vals.csv\", row.names=F);write.csv(train.res$F.mat, \"%s/F_mat.csv\", row.names=F)' | grep '\[1\]'" % (time_stamp, len(L_lst), len(X_lst), time_stamp, time_stamp, time_stamp, self.mu, self.lam, self.gam, self.maxiter, self.tol, time_stamp, time_stamp, time_stamp, time_stamp), shell=True)
        self.model = {
            "G": [np.loadtxt("%s/G_%d.csv" % (time_stamp, i+1), skiprows=1, delimiter=",") for i in range(len(X_lst))],
            "alpha": np.loadtxt("%s/alpha.csv" % time_stamp, skiprows=1, delimiter=","),
            "t": np.loadtxt("%s/vals.csv" % time_stamp, skiprows=1, delimiter=",")[0],
            "diff_G": np.loadtxt("%s/vals.csv" % time_stamp, skiprows=1, delimiter=",")[1],
            "F_mat": np.loadtxt("%s/F_mat.csv" % time_stamp, skiprows=1, delimiter=","),
        }
        call("rm -rf %s/" % time_stamp, shell=True)

    def model_predict(self, test_dataset):
        X_lst, _, _ = self.preprocessing(test_dataset, inf=2)
        Y_lst = [x.dot(self.model["G"][ix]) for ix, x in enumerate(X_lst)]
        Y_lst = [self.model["alpha"][iy]*(y/np.tile(y.sum(axis=1), (y.shape[1],1)).T) for iy, y in enumerate(Y_lst)]
        preds = np.ravel(reduce(sum, Y_lst))
        scores = create_scores(preds, test_dataset)
        return scores