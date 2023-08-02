#coding: utf-8

from stanscofi.models import BasicModel
from stanscofi.preprocessing import preprocessing_XY

from pyffm import PyFFM ## FM not tested!
import pandas as pd
import numpy as np

class FFMWrapper(BasicModel):
    def __init__(self, params=None):
        params = params if (params is not None) else self.default_parameters()
        super(FFMWrapper, self).__init__(params)
        assert self.preprocessing_str in ["Perlman_procedure", "meanimputation_standardize", "same_feature_preprocessing"]
        self.model = PyFFM(model='ffm', 
            training_params={"epochs": self.epochs, "reg_lambda": self.reg_lambda}
        )
        self.scalerS, self.scalerP, self.filter = None, None, None
        self.name = "FFMWrapper"

    def default_parameters(self):
        params = {
            "epochs": 2, #5,
            "reg_lambda": 0.002,
            "preprocessing_str": "meanimputation_standardize", "subset": None,
        }
        return params

    def preprocessing(self, dataset, is_training=True):
        X, y, scalerS, scalerP, filter_ = preprocessing_XY(dataset, self.preprocessing_str, subset_=self.subset, filter_=self.filter, scalerS=self.scalerS, scalerP=self.scalerP, inf=2, njobs=1)
        self.filter = filter_
        self.scalerS = scalerS
        self.scalerP = scalerP
        keep_ids = (y!=0)
        df = pd.DataFrame(np.concatenate((np.asarray(y[keep_ids]>0, dtype=int).reshape((np.sum(keep_ids),1)), X[keep_ids,:]), axis=1), index=range(np.sum(keep_ids)), columns=["click"]+list(map(str,range(X.shape[1]))))
        return [df] if (is_training) else [df, keep_ids]

    def model_fit(self, df):
        self.model.train(df, label_name="click")

    def model_predict_proba(self, df, keep_ids):
        preds = self.model.predict(df.drop(columns=["click"]))
        scores = np.zeros(keep_ids.shape)
        scores[keep_ids] = preds
        return scores
