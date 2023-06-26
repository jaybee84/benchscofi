#coding: utf-8

from stanscofi.models import BasicModel
from benchscofi.utils import utils

import rpy2
import rpy2.robjects as robjects

class LRSSL(BasicModel):
    def __init__(self, params=None):
        r = robjects.r
        r.source('LRSSL.R')
        r.rfunc(folder)
        params = params if (params is not None) else self.default_parameters()
        super(PulearnWrapper, self).__init__(params)
        assert self.classifier in ["ElkanotoPuClassifier","WeightedElkanotoPuClassifier","BaggingPuClassifier"]
        self.scalerS, self.scalerP, self.filter = None, None, None
        self.name = self.classifier ## TODO change name if needed

    def default_parameters(self):
        params = {
            "decision_threshold": 1, 
            "classifier_params": {
                "estimator": sklearn.SVC(C=10, kernel='rbf', gamma=0.4, probability=True),
                "hold_out_ratio": 0.2,
                #"labeled":10, "unlabeled":20, "n_estimators":15,
            },
            "classifier": "ElkanotoPuClassifier",
            "preprocessing": "meanimputation_standardize",
        }
        return params

    def preprocessing(self, dataset):
        X, y, scalerS, scalerP, filter_ = utils.preprocessing_routine(dataset, self.preprocessing_str, subset_=self.subset, filter_=self.filter, scalerS=self.scalerS, scalerP=self.scalerP, inf=2, njobs=1)
        self.filter = filter_
        self.scalerS = scalerS
        self.scalerP = scalerP
        return X, y
        
    def fit(self, train_dataset):
        self.model = eval("pulearn."+self.classifier)(self.classifier_params)
        X, y = self.preprocessing(train_dataset)
        self.model.fit(X, y)

    def model_predict(self, test_dataset):
        X, _ = self.preprocessing(test_dataset)
        preds = self.model.predict(X)
        scores = utils.create_scores(preds, test_dataset)
        return scores
