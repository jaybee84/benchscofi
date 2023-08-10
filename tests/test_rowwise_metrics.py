import unittest
import stanscofi.datasets
import stanscofi.utils
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
        model.fit(dataset)

        scores = model.predict_proba(dataset) # ev=12 row-wise AUC

        [mat, rats] = model.preprocessing(dataset, is_training=False)
        _ = model.model_predict_proba(mat, rats, ev=13, rm=False) # ev=13 column-wise AUC

        print("\n\n"+('_'*27)+"\nMODEL "+model.name)

        rowwise_aucs = rowwise_metrics.calc_auc(scores, dataset) ## row-wise
        print("item-averaged (row) AUC %.3f" % np.mean(rowwise_aucs))
        colwise_aucs = rowwise_metrics.calc_auc(scores, dataset, transpose=True) ## column-wise
        print("user-averaged (column) AUC %.3f" % np.mean(colwise_aucs))

        _, rowwise_aucs, _, row_auc = rowwise_metrics.calc_mpr_auc(scores, dataset) ## row-wise
        print("item-averaged (row) AUC %.3f" % row_auc)
        _, colwise_aucs, _, col_auc = rowwise_metrics.calc_mpr_auc(scores, dataset, transpose=True) ## column-wise
        print("user-averaged (column) AUC %.3f" % col_auc)

        print(("_"*27)+"\n\n")
        ## if it ends without any error, it is a success

if __name__ == '__main__':
    unittest.main()