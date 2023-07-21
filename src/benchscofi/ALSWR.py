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
        self.model = AlternatingLeastSquares.ALSWR(**{(k if (k!="random_state") else "seed"):params[k] for k in params})

    def default_parameters(self):
        params = AlternatingLeastSquares.alswr_params
        params.update({"random_state": 1354})
        return params

    def preprocessing(self, dataset, is_training=True):
        ## users x items: only 1's and -1's
        if (is_training):
            ratings = csr_matrix((dataset.ratings.data, (dataset.ratings.col, dataset.ratings.row)), shape=dataset.ratings.T.shape)
        else:
            ids = np.argwhere(np.ones(dataset.ratings.shape))
            ratings = csr_matrix((dataset.ratings.toarray().ravel(), (ids[:,1].ravel(), ids[:,0].ravel())), shape=dataset.ratings.T.shape)
        return [ratings]
        
    def model_fit(self, X_train):
        np.random.seed(self.random_state)
        self.model.fit(X_train)

    def model_predict_proba(self, X_test):
        preds = self.model.predict().T
        return preds