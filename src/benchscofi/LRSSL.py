#coding: utf-8

from stanscofi.models import BasicModel
from stanscofi.preprocessing import CustomScaler
from benchscofi.utils import tools

import numpy as np
from subprocess import call

class LRSSL(BasicModel):
    def __init__(self, params=None):
        params = params if (params is not None) else self.default_parameters()
        super(LRSSL, self).__init__(params)
        self.scalerS, self.scalerP = None, None
        self.sep_feature = params.get("sep_feature", "-")
        self.name = "LRSSL"
        self.model = None
        self.k, self.mu, self.lam, self.gam, self.maxiter, self.tol = [params[p] for p in ["k","mu","lam","gam","maxiter","tol"]]

    def default_parameters(self):
        params = {
            "decision_threshold": 1, 
            "k": 10, "mu": 0.01, "lam": 0.01, "gam": 2, "tol": 0.01, "maxiter": 1000, 
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
        if (all([self.sep_feature in f for f in dataset.item_features])):
            types_feature = [f.split(self.sep_feature)[0] for f in dataset.item_features]
            X_lst = [S_[:,np.argwhere(np.array(types_feature)==tf)] for tf in list(set(types_feature))]
            #X_lst = [x if (x.shape[0]==x.shape[1]) else np.corrcoef(x) for x in X_lst] ## build a correlation matrix
            X_p = P_ if (P_.shape[0]==P_.shape[1]) else np.corrcoef(P_)
        else:
            X_lst = [S_]
            #X_lst = [S_ if (S_.shape[0]==S_.shape[1]) else np.corrcoef(S_)] ## build a correlation matrix
            X_p = P_ if (P_.shape[0]==P_.shape[1]) else np.corrcoef(P_)
        S_lst = [x.T.dot(x)/np.sqrt(x.sum(axis=0).dot(x.sum(axis=0).T)) for x in X_lst]
        S_lst = [s-np.diag(np.diag(s)) for s in S_lst]
        Y = dataset.ratings_mat.copy()
        ST = np.zeros((Y.shape[0], Y.shape[0]))
        for i in range(Y.shape[0]):
            for j in range(i+1,Y.shape[0]):
                ST[i,j] = np.max(X_p[Y[i,:]==1,:][:,Y[j,:]==1])
        ST = ST + ST.T
        S_lst.append(ST)
        return X_lst, S_lst, Y
        
    def fit(self, train_dataset):
        X_lst, S_lst, Y = self.preprocessing(train_dataset, inf=2)
        cmd = "wget -qO - \'https://raw.githubusercontent.com/LiangXujun/LRSSL/master/LRSSL.R\' | sed -n \'/###/q;p\' > LRSSL.R"
        call(cmd, shell=True)
        L_lst = []
        for s in S_lst:
            np.savetxt("s.csv",s)
            call("R -q -e 'source(\"LRSSL.R\");S <- as.matrix(read.table(\"s.csv\", sep=\" \", header=F));Sp <- get.knn.graph(S, %d);try(write.csv(S, \"s.csv\", row.names=F, header=F), silent=T)' 2>&1 >/dev/null" % self.k, shell=True)
            L_lst.append(np.loadtxt("s.csv"))
            call("rm -f s.csv", shell=True)
        L_lst = [np.diag(L.sum(axis=0))-L for L in L_lst]
        for i, x in enumerate(X_lst):
            np.savetxt("x_%d.csv" % (i+1),x)
        for i, l in enumerate(L_lst):
            np.savetxt("l_%d.csv" % (i+1),l)
        call("R -q -e 'source(\"LRSSL.R\");ml <- %d;mx <- %d;train.res <- as.matrix(read.table(\"s.csv\", sep=\" \", header=F));train.res <- lrssl(X_lst, L_lst, Y, mx, ml, %f, %f, %f, %d, %f);try(write.csv(S, \"s.csv\", row.names=F, header=F), silent=T)'" % (len(L_lst), len(X_lst), self.mu, self.lam, self.gam, self.maxiter, self.tol), shell=True)
        self.model = r.lrssl(X_lst, L_lst, Y, len(X_lst), len(L_lst), self.mu, self.lam, self.gam, self.maxiter, self.tol)
        call("rm -f LRSSL.R", shell=True)

    def model_predict(self, test_dataset):
        X_lst, S_lst, Y = self.preprocessing(test_dataset, inf=2)
        X, _ = self.preprocessing(test_dataset)
        preds = self.model.predict(X)
        scores = utils.create_scores(preds, test_dataset)
        return scores
