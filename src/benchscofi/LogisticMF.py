#coding: utf-8

from stanscofi.models import BasicModel
import numpy as np

from benchscofi.implementations import LogisticMatrixFactorization

class LogisticMF(BasicModel):
    def __init__(self, params=None):
        params = params if (params is not None) else self.default_parameters()
        super(LogisticMF, self).__init__(params)
        self.name = "LogisticMF"
        self.estimator = LogisticMatrixFactorization.LogisticMF(**params)

    def default_parameters(self):
        params = {"counts": np.zeros((5,6)), "num_factors": 2, "reg_param":0.6, "gamma":1.0, "iterations":30}
        return params

    def preprocessing(self, dataset, is_training=True):
        counts = dataset.ratings.toarray().T
        counts[counts<1] = 0
        total = np.sum(counts)
        num_zeros = np.prod(counts.shape)-total
        alpha = num_zeros / total
        #print('alpha %.2f' % alpha)
        counts *= alpha
        return [counts]
        
    def model_fit(self, X):
        self.estimator.counts = X
        self.estimator.num_users = X.shape[0]
        self.estimator.num_items = X.shape[1]
        self.estimator.train_model()

    def model_predict_proba(self, X):
        ## prediction(i,j) = sigmoid(x_i . y_j^T + b_i + b_j)
        fx = self.estimator.user_vectors.dot(self.estimator.item_vectors.T)
        fx += np.tile(self.estimator.user_biases, (1,self.estimator.item_biases.shape[0]))
        fx += np.tile(self.estimator.item_biases, (1,self.estimator.user_biases.shape[0])).T
        preds = np.exp(fx)/(1+np.exp(fx))
        return preds.T





