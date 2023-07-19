#coding: utf-8

from stanscofi.models import BasicModel
import numpy as np
from scipy.sparse import csr_matrix

from benchscofi.implementations import AlternatingLeastSquares

class ALSWR(BasicModel):
    def __init__(self, params=None):
        params = params if (params is not None) else self.default_parameters()
        super(ALSWR, self).__init__(params)
        self.name = "ALSWR"
        self.model = AlternatingLeastSquares.ALSWR(**{(k if (k!="random_state") else "seed"):params[k] for k in params if (k not in ["decision_threshold"])})

    def default_parameters(self):
        params = AlternatingLeastSquares.alswr_params
        params.update({"random_state": 1354})
        return params

    def preprocessing(self, dataset, is_training=True):
        ## users x items
        ratings = csr_matrix((np.array(dataset.ratings[:,2], dtype=np.float64), (dataset.ratings[:,0], dataset.ratings[:,0])), shape=(dataset.ratings_mat.shape[1],dataset.ratings_mat.shape[0]))
        return [ratings]
        
    def model_fit(self, X_train):
        np.random.seed(self.random_state)
        self.model.fit(X_train)

    def model_predict_proba(self, X_test):
        return self.model.predict()