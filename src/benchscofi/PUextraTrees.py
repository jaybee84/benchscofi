#coding: utf-8

from stanscofi.models import BasicModel
from stanscofi.preprocessing import preprocessing_XY
import numpy as np
from multiprocessing import cpu_count

## https://github.com/jonathanwilton/PUExtraTrees/blob/2aaa506309ccc732d414c3b2522bf0320a327497
from benchscofi.implementations import PUextraTreeImplementation

class PUextraTrees(BasicModel):
    def __init__(self, params=None):
        params_ = self.default_parameters()
        if (params is not None):
            params_.update(params)
        super(PUextraTrees, self).__init__(params_)
        assert self.pi<1 and self.pi>0
        assert self.risk_estimator=="nnPU"
        assert self.loss in ["quadratic", "logistic"]
        assert self.n_jobs>0 and self.n_jobs<cpu_count()+1
        self.name = "PUextraTrees"
        self.estimator = None
        self.filter = params_.get("filter")
        self.scalerS = None
        self.scalerP = None

    def default_parameters(self):
        params = dict(
        	pi = 0.10,
        	n_estimators = 10,#100,
		risk_estimator = "nnPU",
		loss = "quadratic", # "logistic"
		max_depth = None, 
		min_samples_leaf = 1, 
		max_features = "sqrt", 
		max_candidates = 1,
		n_jobs = min(cpu_count()-2,10),
        	filter_ = None,
        	preprocessing_str = "meanimputation_standardize",
        	subset = None,
        )
        return params

    def preprocessing(self, dataset, is_training=True):
        X, y, scalerS, scalerP, filter_ = preprocessing_XY(dataset, self.preprocessing_str, subset_=self.subset, filter_=self.filter, scalerS=self.scalerS, scalerP=self.scalerP, inf=2, njobs=self.n_jobs)
        y = y.astype(np.int8)
        X = np.nan_to_num(X, nan=0.)
        self.filter = filter_
        self.scalerS = scalerS
        self.scalerP = scalerP
        U = X.copy()
        P = X[y==1,:]
        return [P, U] if (is_training) else [X]

    def model_fit(self, P, U):
        self.estimator = PUextraTreeImplementation.PUExtraTrees(n_estimators=self.n_estimators,risk_estimator=self.risk_estimator,loss=self.loss,max_depth=self.max_depth,min_samples_leaf=self.min_samples_leaf,max_features=self.max_features,max_candidates=self.max_candidates,n_jobs=self.n_jobs)
        self.estimator.fit(P=P, U=U, pi=self.pi)

    def model_predict_proba(self, X):
    	return self.estimator.predict(X).ravel()
