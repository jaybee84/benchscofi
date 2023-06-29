#coding: utf-8

## http://bioinformatics.csu.edu.cn/resources/softs/DrugRepositioning/DRRS/index.html
from stanscofi.models import BasicModel, create_overscores
from stanscofi.preprocessing import CustomScaler

import numpy as np
from subprocess import call

import calendar
import time
current_GMT = time.gmtime()

class DRRS(BasicModel):
    def __init__(self, params=None):
        params = params if (params is not None) else self.default_parameters()
        super(DRRS, self).__init__(params)
        self.scalerS, self.scalerP = None, None
        self.name = "DRRS" 
        self.model = None
        self.use_masked_dataset = False

    def default_parameters(self):
        params = {
            "decision_threshold": 1, 
            "preprocessing": "meanimputation_standardize",
        }
        return params

    def preprocessing(self, dataset):
        if (self.scalerS is None):
            self.scalerS = CustomScaler(posinf=inf, neginf=-inf)
        S_ = self.scalerS.fit_transform(dataset.items.T.copy(), subset=None)
        X_s = S_ if (S_.shape[0]==S_.shape[1]) else np.corrcoef(S_)
        if (self.scalerP is None):
            self.scalerP = CustomScaler(posinf=inf, neginf=-inf)
        P_ = self.scalerP.fit_transform(dataset.users.T.copy(), subset=None)
        X_p = P_ if (P_.shape[0]==P_.shape[1]) else np.corrcoef(P_)
        A_sp = dataset.ratings_mat
        return X, y
        
    def fit(self, train_dataset):
        self.model = eval("pulearn."+self.classifier)(self.classifier_params)
        X, y = self.preprocessing(train_dataset)
        self.model.fit(X, y)

    def model_predict(self, test_dataset):
        assert test_dataset.folds is not None
        preds = self.model.imputed_A
        df = test_dataset.folds
        scores = utils.create_overscores(df, test_dataset)
        return scores
