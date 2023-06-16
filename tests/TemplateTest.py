import unittest
import numpy as np
import sys

import stanscofi.datasets
import stanscofi.training_testing
import stanscofi.validation

sys.path.insert(0, "../src/")
import benchscofi
import benchscofi.XXXXXX

class TestModel(unittest.TestCase):

    ## Generate example
    def generate_dataset(self, random_seed):
        npositive, nnegative, nfeatures, mean, std = 200, 100, 50, 0.5, 1
        data_args = stanscofi.datasets.generate_dummy_dataset(npositive, nnegative, nfeatures, mean, std, random_state=random_seed)
        dataset = stanscofi.datasets.Dataset(**data_args)
        return dataset

    ## Test whether basic functions of the model work
    def test_model(self): 
        random_seed = 124565 
        np.random.seed(random_seed)
        dataset = self.generate_dataset(random_seed)
        test_size = 0.3
        train_set, test_set, _, _ = stanscofi.training_testing.traintest_validation_split(dataset, test_size, early_stop=2, metric="euclidean", disjoint_users=False, random_state=random_seed, verbose=False, print_dists=False)
        train_dataset = dataset.get_folds(train_set)
        test_dataset = dataset.get_folds(test_set)
        model = benchscofi.XXXXXX.XXXXXX()
        model.fit(train_dataset)
        scores = model.predict(test_dataset)
        print("\n\n"+('_'*27)+"\nMODEL "+model.name)
        model.print_scores(scores)
        predictions = model.classify(scores)
        model.print_classification(predictions)
        metrics, _ = stanscofi.validation.compute_metrics(scores, predictions, test_dataset, beta=1, verbose=False)
        print(metrics)
        print(("_"*27)+"\n\n")
        ## if it ends without any error, it is a success