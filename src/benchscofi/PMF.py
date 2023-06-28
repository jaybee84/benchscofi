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
        self.random_state = params["random_state"]
        self.name = "PMF"
        self.model = BayesianPairwiseRanking.BPR(**{k:params[k] for k in params if (k not in ["random_state","decision_threshold"])})

    def default_parameters(self):
        params = BayesianPairwiseRanking.bpr_params
        params.update({"random_state": 1354, "decision_threshold": 1})
        return params

    def preprocessing(self, dataset):
        return csr_matrix(dataset.ratings_mat.T)
        
    def fit(self, train_dataset):
        np.random.seed(self.random_state)
        Y = self.preprocessing(train_dataset)
        self.model.fit(Y)

    def model_predict(self, test_dataset):
        assert test_dataset.folds is not None
        ids = np.argwhere(np.ones(test_dataset.ratings_mat.shape))
        preds = np.zeros((ids.shape[0], 3))
        preds[:,0] = ids[:,1]
        preds[:,1] = ids[:,0]
        in_fold = [((test_dataset.folds[:,1]==i)&(test_dataset.folds[:,0]==j)).any() for i,j in ids[:,:2].tolist()]
        assert sum(in_fold)==test_dataset.folds.shape[0]
        preds[in_fold,2] = [self.model._predict_user(user)[item] for user, item in test_dataset.folds[:,:2].tolist()]
        return preds
