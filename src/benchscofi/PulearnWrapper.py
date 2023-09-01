#coding: utf-8

from stanscofi.models import BasicModel
from stanscofi.preprocessing import preprocessing_XY
import numpy as np

 ## UNLABELED ARE -1 POSITIVE ARE 1
import pulearn

class PulearnWrapper(BasicModel):
    def __init__(self, params=None):
        params = params if (params is not None) else self.default_parameters()
        super(PulearnWrapper, self).__init__(params)
        assert self.classifier in ["ElkanotoPuClassifier","WeightedElkanotoPuClassifier","BaggingPuClassifier"]
        self.scalerS, self.scalerP, self.filter = None, None, None
        self.name = self.classifier
        self.estimator = eval("pulearn."+self.classifier)(**self.classifier_params)

    def default_parameters(self):
        from sklearn.svm import SVC
        params = {
            "classifier_params": {
                "estimator": SVC(C=10, kernel='rbf', gamma=0.4, probability=True),
                "hold_out_ratio": 0.2,
                #"labeled":10, "unlabeled":20, "n_estimators":15,
            },
            "classifier": "ElkanotoPuClassifier",
            "preprocessing_str": "meanimputation_standardize",
            "subset": None,
        }
        return params

    def preprocessing(self, dataset, is_training=True):
        X, y, scalerS, scalerP, filter_ = preprocessing_XY(dataset, self.preprocessing_str, subset_=self.subset, filter_=self.filter, scalerS=self.scalerS, scalerP=self.scalerP, inf=2, njobs=1)
        self.filter = filter_
        self.scalerS = scalerS
        self.scalerP = scalerP
        ## Unlabeled samples are -1 in pulearn
        y[y<1] = -1
        return [X, y] if (is_training) else [X]
        
    def model_fit(self, X, y):
        self.estimator.fit(X, y)

    def model_predict_proba(self, X):
        preds = self.estimator.predict_proba(X)[:,int(np.argwhere(self.estimator.classes_==1))]
        return preds
