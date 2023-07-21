#coding: utf-8

from stanscofi.models import BasicModel
import numpy as np
from scipy.sparse import csr_matrix

from benchscofi.implementations import BayesianPairwiseRanking

#' Matrix factorization
class PMF(BasicModel):
    def __init__(self, params=None):
        params = params if (params is not None) else self.default_parameters()
        super(PMF, self).__init__(params)
        self.name = "PMF"
        self.estimator = BayesianPairwiseRanking.BPR(**params)

    def default_parameters(self):
        params = BayesianPairwiseRanking.bpr_params
        return params

    def preprocessing(self, dataset, is_training=True):
        ## Use 0-1 ratings (matrix form)
        return [csr_matrix(dataset.ratings.toarray().T)]
        
    def model_fit(self, X):
        self.estimator.fit(X)

    def model_predict_proba(self, X):
        preds = self.estimator.predict().T
        return preds
