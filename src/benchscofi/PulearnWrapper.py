#coding: utf-8

from stanscofi.models import BasicModel, create_scores
from stanscofi.preprocessing import preprocessing_routine
import numpy as np

import pulearn

class PulearnWrapper(BasicModel):
    def __init__(self, params=None):
        params = params if (params is not None) else self.default_parameters()
        assert all([x in params for x in ["random_state", "classifier", "classifier_params", "preprocessing", "subset"]])
        assert params["classifier"] in ["ElkanotoPuClassifier","WeightedElkanotoPuClassifier","BaggingPuClassifier"]
        self.classifier = params["classifier"]
        self.random_state = params["random_state"]
        self.classifier_params = params["classifier_params"]
        self.subset = params.get("subset")
        self.preprocessing_str = params["preprocessing"] 
        super(PulearnWrapper, self).__init__(params)
        self.scalerS, self.scalerP, self.filter = None, None, None
        self.name = self.classifier

    def default_parameters(self):
        from sklearn.svm import SVC
        params = {
            "decision_threshold": 1, 
            "classifier_params": {
                "estimator": SVC(C=10, kernel='rbf', gamma=0.4, probability=True),
                "hold_out_ratio": 0.2,
                #"labeled":10, "unlabeled":20, "n_estimators":15,
            },
            "classifier": "ElkanotoPuClassifier",
            "preprocessing": "meanimputation_standardize",
            "subset": None,
            "random_state": 124565,
        }
        return params

    def preprocessing(self, dataset):
        X, y, scalerS, scalerP, filter_ = preprocessing_routine(dataset, self.preprocessing_str, subset_=self.subset, filter_=self.filter, scalerS=self.scalerS, scalerP=self.scalerP, inf=2, njobs=1)
        self.filter = filter_
        self.scalerS = scalerS
        self.scalerP = scalerP
        return X, y
        
    def fit(self, train_dataset):
        np.random.seed(self.random_state)
        self.model = eval("pulearn."+self.classifier)(**self.classifier_params)
        X, y = self.preprocessing(train_dataset)
        self.model.fit(X, y)

    def model_predict(self, test_dataset):
        X, _ = self.preprocessing(test_dataset)
        preds = self.model.predict(X)
        scores = create_scores(preds, test_dataset)
        return scores
