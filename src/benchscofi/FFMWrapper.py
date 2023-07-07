#coding: utf-8

from stanscofi.models import BasicModel, create_scores
from stanscofi.preprocessing import preprocessing_routine

from pyffm import PyFFM ## FM not tested!
import pandas as pd
import numpy as np

class FFMWrapper(BasicModel):
    def __init__(self, params=None):
        params = params if (params is not None) else self.default_parameters()
        super(FFMWrapper, self).__init__(params)
        self.epochs = params["epochs"] 
        self.reg_lambda = params["reg_lambda"]
        self.preprocessing_str = params["preprocessing"]
        self.subset = params["subset"]
        assert self.preprocessing_str in ["Perlman_procedure", "meanimputation_standardize", "same_feature_preprocessing"]
        self.model = PyFFM(model='ffm', 
            training_params={"epochs": self.epochs, "reg_lambda": self.reg_lambda}
        )
        self.scalerS, self.scalerP, self.filter = None, None, None
        self.name = "FFMWrapper"
        self.use_masked_dataset = False

    def default_parameters(self):
        params = {
            "decision_threshold": 0, 
            "random_state": 1245,
            "epochs": 2, 
            "reg_lambda": 0.002,
            "preprocessing": "meanimputation_standardize", "subset": None,
        }
        return params

    def preprocessing(self, dataset):
        X, y, scalerS, scalerP, filter_ = preprocessing_routine(dataset, self.preprocessing_str, subset_=self.subset, filter_=self.filter, scalerS=self.scalerS, scalerP=self.scalerP, inf=2, njobs=1)
        self.filter = filter_
        self.scalerS = scalerS
        self.scalerP = scalerP
        keep_ids = (y!=0)
        df = pd.DataFrame(np.concatenate((np.asarray(y[keep_ids]>0, dtype=int).reshape((np.sum(keep_ids),1)), X[keep_ids,:]), axis=1), index=range(np.sum(keep_ids)), columns=["click"]+list(map(str,range(X.shape[1]))))
        return df, keep_ids

    def fit(self, train_dataset):
        df, _ = self.preprocessing(train_dataset)
        self.model.train(df, label_name="click")

    def predict(self, test_dataset):
        df, keep_ids = self.preprocessing(test_dataset)
        ids = np.argwhere(np.ones(test_dataset.ratings_mat.shape))
        preds = np.asarray(np.zeros((ids.shape[0],3)), dtype=int)
        preds[:,0] = ids[:,1]
        preds[:,1] = ids[:,0]
        preds[keep_ids,2] = self.model.predict(df.drop(columns=["click"]))
        return preds
