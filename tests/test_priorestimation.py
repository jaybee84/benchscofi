import unittest
from glob import glob
from subprocess import call
import sys
import stanscofi.datasets
import numpy as np
from stanscofi.preprocessing import meanimputation_standardize
from stanscofi.training_testing import random_simple_split

import sys ##
sys.path.insert(0, "../src/") ##
import benchscofi
from benchscofi.utils import prior_estimation
from benchscofi.PulearnWrapper import PulearnWrapper

def compute_naive_estimators(dataset, labels_mat):
    pos, neg = np.sum(dataset.ratings.data>0), np.sum(dataset.ratings.data<0)
    known, total = np.sum(dataset.ratings.data!=0), np.prod(dataset.ratings.shape)
    pos_total = pos/total
    sparsity = known/total
    imbalance = neg/pos
    pi = np.nan if(labels_mat is None) else np.sum(labels_mat.values>0)/np.prod(labels_mat.values.shape)
    c = pos_total/pi if (labels_mat is not None) else np.nan
    return sparsity, pi, c, imbalance

class TestUtils(unittest.TestCase):

    def test_generate_Censoring_dataset(self):
        synthetic_params = {
            "N":100, "nfeatures":10, "mean":2, "std":0.1, "exact": True, 
            "pi": 0.3, "c": 0.2
        }
        data_args, labels_mat = prior_estimation.generate_Censoring_dataset(**synthetic_params)
        dataset = stanscofi.datasets.Dataset(**data_args)
        _, pi, c, _ = compute_naive_estimators(dataset, labels_mat)
        self.assertEqual(np.round(pi,1), synthetic_params["pi"])
        self.assertEqual(np.round(c,1), synthetic_params["c"])

    def test_generate_CaseControl_dataset(self):
        synthetic_params = {
            "N":100, "nfeatures":10, "mean":2, "std":0.1, "exact": True, 
            "pi": 0.3, "imbalance": 0.0, "sparsity": 0.1
        }
        data_args, labels_mat = prior_estimation.generate_CaseControl_dataset(**synthetic_params)
        dataset = stanscofi.datasets.Dataset(**data_args)
        sparsity, pi, _, imbalance = compute_naive_estimators(dataset, labels_mat)
        self.assertEqual(np.round(pi,1), synthetic_params["pi"])
        self.assertEqual(np.round(sparsity,1), synthetic_params["sparsity"])
        self.assertEqual(np.round(imbalance,3), synthetic_params["imbalance"])

    def test_data_aided_estimation(self): 
        synthetic_params = {
             "N":10000, "nfeatures":10, "mean":2, "std":0.1, "exact": True, 
            "pi": 0.3, "c": 0.2
        }
        data_args, labels_mat = prior_estimation.generate_Censoring_dataset(**synthetic_params)
        dataset = stanscofi.datasets.Dataset(**data_args)
        (traintest_folds, val_folds), _ = random_simple_split(dataset, 0.2, metric="euclidean")
        traintest_dataset = dataset.subset(traintest_folds, subset_name="Train Test")
        val_dataset = dataset.subset(val_folds, subset_name="Validation")
        model = PulearnWrapper()
        model.fit(traintest_dataset)
        scores_test = model.predict_proba(val_dataset).toarray().ravel()
        pred_scores = np.array([max(min(s,1),0) for s in scores_test])
        y_test = (val_dataset.folds.toarray()*val_dataset.ratings.toarray()).ravel()
        for est_type in [1,2,3]:
            est = prior_estimation.data_aided_estimation(pred_scores, y_test, estimator_type=est_type)
        ## if it ends without any error, it is a success

    def test_roc_aided_estimation(self): 
        synthetic_params = {
            "N":100, "nfeatures":10, "mean":2, "std":0.1, "exact": True, 
            "pi": 0.3, "imbalance": 0.05, "sparsity": 0.1
        }
        data_args, labels_mat = prior_estimation.generate_CaseControl_dataset(**synthetic_params)
        dataset = stanscofi.datasets.Dataset(**data_args)
        (traintest_folds, val_folds), _ = random_simple_split(dataset, 0.2, metric="euclidean")
        traintest_dataset = dataset.subset(traintest_folds, subset_name="Train Test")
        val_dataset = dataset.subset(val_folds, subset_name="Validation")
        model = PulearnWrapper()
        model.fit(traintest_dataset)
        scores_test = model.predict_proba(val_dataset).toarray().ravel()
        y_test = (val_dataset.folds.toarray()*val_dataset.ratings.toarray()).ravel()
        y_test[y_test<1] = 0
        for reg_type in [1,2]:
            est_pi = prior_estimation.roc_aided_estimation(scores_test, y_test, regression_type=reg_type)
        ## if it ends without any error, it is a success

    def test_divergence_aided_estimation(self): 
        synthetic_params = {
            "N":100, "nfeatures":10, "mean":2, "std":0.1, "exact": True, 
            "pi": 0.3, "imbalance": 0.05, "sparsity": 0.1
        }
        data_args, labels_mat = prior_estimation.generate_CaseControl_dataset(**synthetic_params)
        dataset = stanscofi.datasets.Dataset(**data_args)
        X, y, _, _ = meanimputation_standardize(dataset)
        for div_type in ["L1-distance","Pearson"]:
            est_pi = prior_estimation.divergence_aided_estimation(X, y, lmb=1, sigma=1., divergence_type=div_type)
        ## if it ends without any error, it is a success

if __name__ == '__main__':
    unittest.main()