#coding: utf-8

from stanscofi.models import BasicModel, create_overscores
import numpy as np
from scipy.sparse import csr_matrix

from benchscofi.implementations import AlternatingLeastSquares

class ALSWR(BasicModel):
    def __init__(self, params=None):
        params = params if (params is not None) else self.default_parameters()
        super(ALSWR, self).__init__(params)
        self.random_state = params["random_state"]
        self.name = "ALSWR"
        self.model = AlternatingLeastSquares.ALSWR(**{(k if (k!="random_state") else "seed"):params[k] for k in params if (k not in ["decision_threshold"])})
        self.use_masked_dataset = True

    def default_parameters(self):
        params = AlternatingLeastSquares.alswr_params
        params.update({"random_state": 1354, "decision_threshold": 1})
        return params

    def preprocessing(self, dataset):
        ## users x items
        ratings = csr_matrix((np.array(dataset.ratings[:,2], dtype=np.float64), (dataset.ratings[:,0], dataset.ratings[:,0])), shape=(dataset.ratings_mat.shape[1],dataset.ratings_mat.shape[0]))
        return ratings
        
    def fit(self, train_dataset):
        np.random.seed(self.random_state)
        X_train = self.preprocessing(train_dataset)
        self.model.fit(X_train)

    def model_predict(self, test_dataset):
        assert test_dataset.folds is not None
        folds = np.copy(test_dataset.folds)
        folds[:,2] = [self.model._predict_user(user)[item] for user, item in folds[:,:2].tolist()]
        scores = create_overscores(folds, test_dataset)
        return scores