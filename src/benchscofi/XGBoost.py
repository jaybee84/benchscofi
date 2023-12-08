#coding: utf-8

from stanscofi.models import BasicModel
from stanscofi.preprocessing import preprocessing_XY
import numpy as np
import xgboost as xgb
from scipy.sparse import coo_array

class XGBoost(BasicModel):
    def __init__(self, params=None):
        params_ = self.default_parameters()
        if (params is not None):
            params_.update(params)
        params = params if (params is not None) else self.default_parameters()
        super(XGBoost, self).__init__(params_)
        assert "binary:"==self.objective[:len("binary:")]
        assert self.eta<=1 and self.eta>=0
        assert self.gamma>=0
        assert self.max_depth>=0
        assert self.min_child_weight>=0
        assert self.max_delta_step>=0
        assert self.sampling_method in ["uniform","gradient_based"]
        assert self.reg_lambda>=0
        assert self.alpha>=0
        self.name = "XGBoost"
        self.estimator = xgb.XGBClassifier(**{p:params[p] for p in params if (p not in ["preprocessing_str","subset"])})
        self.filter = None
        self.scalerS = None
        self.scalerP = None

    def default_parameters(self):
        params = dict(random_state=1234,objective="binary:logistic",n_estimators=100,booster="gbtree",device="cpu",eta=0.001,gamma=0,max_depth=6,min_child_weight=1,max_delta_step=0,sampling_method="uniform",reg_lambda=1,alpha=0,tree_method="hist",preprocessing_str = "meanimputation_standardize", subset=None, n_jobs=1)
        return params

    def preprocessing(self, dataset, is_training=True):
        X, y, scalerS, scalerP, filter_ = preprocessing_XY(dataset, self.preprocessing_str, subset_=self.subset, filter_=self.filter, scalerS=self.scalerS, scalerP=self.scalerP, inf=2, njobs=1)
        y[y<1] = 0 ## no unlabeled points
        X = np.nan_to_num(X, nan=0.)
        self.filter = filter_
        self.scalerS = scalerS
        self.scalerP = scalerP
        return [X,y] if (is_training) else [X,dataset]
        
    def model_fit(self, X, y):
        self.estimator.fit(X, y)

    def model_predict_proba(self, X, dataset):
        return self.estimator.predict(X).ravel()
        
    def predict(self, scores, threshold=0.5):
        '''
        Outputs class labels based on the scores, using the following formula
            prediction = -1 if (score<threshold) else 1

        ...

        Parameters
        ----------
        scores : COO-array of shape (n_items, n_users)
            sparse matrix in COOrdinate format
        threshold : float
            the threshold of classification into the positive class

        Returns
        ----------
        predictions : COO-array of shape (n_items, n_users)
            sparse matrix in COOrdinate format with values in {-1,1}
        '''
        return coo_array(((-1)**(scores.data<=threshold), (scores.row, scores.col)), shape=scores.shape)
