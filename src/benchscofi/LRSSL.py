#coding: utf-8

## https://github.com/LiangXujun/LRSSL/tree/a16a75c028393e7256e3630bc8b7900026061f99
from stanscofi.models import BasicModel
from stanscofi.preprocessing import CustomScaler

import numpy as np
from subprocess import call
from functools import reduce

import calendar
import time
current_GMT = time.gmtime()

class LRSSL(BasicModel):
    def __init__(self, params=None):
        call("R -q -e \"print('R is installed and running.')\"", shell=True)
        self.LRSSL_filepath="LRSSL.R"
        cmd = "wget -qO - \'https://raw.githubusercontent.com/LiangXujun/LRSSL/a16a75c028393e7256e3630bc8b7900026061f99/LRSSL.R\' | sed -n \'/###/q;p\' > %s" % self.LRSSL_filepath
        call(cmd, shell=True)
        params = params if (params is not None) else self.default_parameters()
        super(LRSSL, self).__init__(params)
        self.scalerS, self.scalerP = None, None
        self.name = "LRSSL"
        self.estimator = None

    def default_parameters(self):
        params = {
            "k": 10, "mu": 0.01, "lam": 0.01, "gam": 2, "tol": 1e-2, "maxiter": 500, "sep_feature": "-",
        }
        return params

    def preprocessing(self, dataset, is_training=True, inf=2):
        if (self.scalerS is None):
            self.scalerS = CustomScaler(posinf=inf, neginf=-inf)
        S_ = self.scalerS.fit_transform(dataset.items.T.toarray().copy(), subset=None)
        S_ = np.nan_to_num(S_, nan=0.0) ##
        if (self.scalerP is None):
            self.scalerP = CustomScaler(posinf=inf, neginf=-inf)
        P_ = self.scalerP.fit_transform(dataset.users.T.toarray().copy(), subset=None)
        P_ = np.nan_to_num(P_, nan=0.0) ##
        if (all([self.sep_feature in str(f) for f in dataset.item_features])):
            types_feature = [str(f).split(self.sep_feature)[0] for f in dataset.item_features]
            X_lst = [S_[:,np.argwhere(np.array(types_feature)==tf)].T for tf in list(set(types_feature))]
            X_lst = [x.reshape(x.shape[1:]) for x in X_lst]
        else:
            X_lst = [S_.T]
        X_p = P_ if (P_.shape[0]==P_.shape[1]) else np.corrcoef(P_)
        S_lst = [x.T.dot(x)/np.sqrt(x.sum(axis=0).dot(x.sum(axis=0).T)) for x in X_lst]
        S_lst = [s-np.diag(np.diag(s)) for s in S_lst]
        Y = dataset.ratings.toarray().copy()
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
        return [X_lst, S_lst, Y] if (is_training) else [X_lst, dataset.folds.shape]
        
    def model_fit(self, X_lst, S_lst, Y):
        time_stamp = calendar.timegm(current_GMT)+np.random.choice(range(int(1e8)), size=1)[0]
        filefolder = "LRSSL_%s/" % time_stamp
        call("mkdir -p %s/" % filefolder, shell=True)
        L_lst = []
        for s in S_lst:
            np.savetxt("%s/s.csv" % filefolder,s)
            call("R -q -e 'source(\"%s\");S <- as.matrix(read.table(\"%s/s.csv\", sep=\" \", header=F));Sp <- get.knn.graph(S, %d);write.csv(S, \"%s/s.csv\", row.names=F)' 2>&1 >/dev/null" % (self.LRSSL_filepath, filefolder, self.k, filefolder), shell=True)
            L_lst.append(np.loadtxt("%s/s.csv" % filefolder, skiprows=1, delimiter=","))
        L_lst = [np.diag(L.sum(axis=0))-L for L in L_lst]
        for i, x in enumerate(X_lst):
            np.savetxt("%s/x_%d.csv" % (filefolder, i+1),x)
        for i, l in enumerate(L_lst):
            np.savetxt("%s/l_%d.csv" % (filefolder, i+1),l)
        np.savetxt("%s/y.csv" % filefolder, Y)
        assert all([s.shape[0]==Y.shape[0] and s.shape[1]==Y.shape[0] for s in S_lst])
        assert all([l.shape[0]==Y.shape[0] and l.shape[1]==Y.shape[0] for l in L_lst])
        assert all([x.shape[0]==Y.shape[0] and x.shape[1]==Y.shape[0] for x in X_lst])
        call("R -q -e 'source(\"%s\");ml <- %d;mx <- %d;X_lst <- lapply(1:mx, function(i) as.matrix(read.table(paste0(\"%s/x_\",i,\".csv\"), sep=\" \", header=F)));L_lst <- lapply(1:ml, function(i) as.matrix(read.table(paste0(\"%s/l_\",i,\".csv\"), sep=\" \", header=F)));Y <- as.matrix(read.table(\"%s/y.csv\", sep=\" \", header=F));train.res <- lrssl(X_lst, L_lst, Y, mx, ml, %f, %f, %f, %d, %f);for(i in 1:mx){write.csv(train.res$Gs[[i]], paste0(\"%s/G_\",i,\".csv\"), row.names=F)};write.csv(train.res$alpha, \"%s/alpha.csv\", row.names=F);write.csv(list(t=train.res$t,diff_G=train.res$diff.G), \"%s/vals.csv\", row.names=F);write.csv(train.res$F.mat, \"%s/F_mat.csv\", row.names=F)' | grep '\[1\]'" % (self.LRSSL_filepath, len(L_lst), len(X_lst), filefolder, filefolder, filefolder, self.mu, self.lam, self.gam, self.maxiter, self.tol, filefolder, filefolder, filefolder, filefolder), shell=True)
        self.estimator = {
            "G": [np.loadtxt("%s/G_%d.csv" % (filefolder, i+1), skiprows=1, delimiter=",") for i in range(len(X_lst))],
            "alpha": np.loadtxt("%s/alpha.csv" % filefolder, skiprows=1, delimiter=","),
            "t": np.loadtxt("%s/vals.csv" % filefolder, skiprows=1, delimiter=",")[0],
            "diff_G": np.loadtxt("%s/vals.csv" % filefolder, skiprows=1, delimiter=",")[1],
            "F_mat": np.loadtxt("%s/F_mat.csv" % filefolder, skiprows=1, delimiter=","),
        }
        call("rm -rf %s/ %s" % (filefolder, self.LRSSL_filepath), shell=True)

    def model_predict_proba(self, X_lst, shp):
        Y_lst = [x.dot(self.estimator["G"][ix]) for ix, x in enumerate(X_lst)]
        Y_lst = [self.estimator["alpha"][iy]*(y/np.tile(y.sum(axis=1), (y.shape[1],1)).T) for iy, y in enumerate(Y_lst)]
        preds = np.ravel(reduce(sum, Y_lst))
        return preds.reshape(shp)