import unittest
import stanscofi.datasets
import stanscofi.utils
import stanscofi.validation
import numpy as np
import random
from scipy.sparse import coo_array
from subprocess import Popen

import sys ##
sys.path.insert(0, "../src/") ##
import benchscofi
import benchscofi.LibMFWrapper
from benchscofi.utils import rowwise_metrics

import sys
dataset_name = sys.argv[1] if (len(sys.argv)>1) else ""
sys.argv = sys.argv[:1]

class TestRowwiseMetrics(unittest.TestCase):

    ## Generate example
    def generate_dataset(self, random_seed):
        npositive, nnegative, nfeatures, mean, std = 200, 100, 50, 0.5, 1
        data_args = stanscofi.datasets.generate_dummy_dataset(npositive, nnegative, nfeatures, mean, std, random_state=random_seed) 
        dataset = stanscofi.datasets.Dataset(**data_args)
        return dataset

    ## Load existing dataset
    def load_dataset(self, dataset_name):
        Popen("mkdir -p ../datasets/".split(" "))
        dataset = stanscofi.datasets.Dataset(**stanscofi.utils.load_dataset(dataset_name, "../datasets/"))
        return dataset

    ## Test whether basic functions of the model work
    def test_model(self): 
        random_seed = 124565 
        np.random.seed(random_seed)
        random.seed(random_seed)
        if (len(dataset_name)==0):
            dataset = self.generate_dataset(random_seed)
        else:
            dataset = self.load_dataset(dataset_name)
        model = benchscofi.LibMFWrapper.LibMFWrapper()
        [mat, rats] = model.preprocessing(dataset, is_training=True)
        model.model_fit(mat, rats, rm=False)

        print("\n\n"+('_'*27)+"\nMODEL "+model.name)

        [mat, rats] = model.preprocessing(dataset, is_training=False)
        _ = model.model_predict_proba(mat, rats, ev=13, rm=False)

        print("\n* Metrics from ndcg.py code")

        benchscofi.LibMFWrapper.execute_ndcg(
            model.libmf_folder+"mat.tr.txt", 
            model.libmf_folder+"mat.txt",
            [model.libmf_folder+"ocmf_output.txt"]
        )

        print("\n* Metrics from libMF")

        [mat, rats] = model.preprocessing(dataset, is_training=False)
        _ = model.model_predict_proba(mat, rats, ev=13, rm=False) # ev=13 column-wise AUC
        scores = model.predict_proba(dataset) # ev=12 row-wise AUC

        print("\n* Metrics from rowwise_metrics.calc_auc")

        rowwise_aucs = rowwise_metrics.calc_auc(scores, dataset) ## row-wise
        print("item-averaged (row) AUC %.4f" % np.mean(rowwise_aucs))
        colwise_aucs = rowwise_metrics.calc_auc(scores, dataset, transpose=True) ## column-wise
        print("user-averaged (column) AUC %.4f" % np.mean(colwise_aucs))

        print("\n* Metrics from rowwise_metrics.calc_mpr_auc")

        _, rowwise_aucs, _, row_auc = rowwise_metrics.calc_mpr_auc(scores, dataset) ## row-wise
        print("item-averaged (row) AUC %.4f" % row_auc)
        _, colwise_aucs, _, col_auc = rowwise_metrics.calc_mpr_auc(scores, dataset, transpose=True) ## column-wise
        print("user-averaged (column) AUC %.4f" % col_auc)

        print("\n* Metrics from stanscofi")
        predictions = np.zeros(scores.shape)
        predictions[scores.row, scores.col] = (-1)**(scores.data<0)
        predictions = coo_array(predictions)
        metrics, _ = stanscofi.validation.compute_metrics(scores, predictions, dataset, metrics=["AUC"], k=1, beta=1, verbose=False)
        print("user-averaged AUC %.4f" % metrics.values[0,0])

        print(("_"*27)+"\n\n")
        ## if it ends without any error, it is a success

if __name__ == '__main__':
    unittest.main()