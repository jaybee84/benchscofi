#coding: utf-8

from stanscofi.models import BasicModel
import numpy as np
import libmf
from libmf import mf
from multiprocessing import cpu_count

class LibMF(BasicModel):
    def __init__(self, params=None):
        params = params if (params is not None) else self.default_parameters()
        super(LibMF, self).__init__(params)
        self.name = "LibMF"
        self.model = mf.MF(params)

    def default_parameters(self):
        params = {
            "fun": 0,
            "k": 8,
            "nr_threads": max(cpu_count()-2,1),
            "nr_bins": 26,
            "nr_iters": 20,
            "lambda_p1": 0.04,
            "lambda_p2": 0.0,
            "lambda_q1": 0.04,
            "lambda_q2": 0.0,
            "eta": 0.1,
            "do_nmf": False,
            "quiet": False,
            "copy_data": True,
        }
        return params

    def preprocessing(self, dataset, is_training=True):
        row, col = dataset.folds.row, dataset.folds.col
        rats = dataset.ratings.toarray()[row, col].ravel()
        return [np.vstack(tuple([row, col, rats])).transpose().copy()] if (is_training) else [np.vstack(tuple([row, col])).transpose().copy()]

    def model_fit(self, data):
        self.model.fit(data)

    def model_predict_proba(self, data):
        return self.model.predict(data)