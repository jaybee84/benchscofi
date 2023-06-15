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
    def generate_dataset(self):
        npositive, nnegative, nfeatures, mean, std = 200, 100, 50, 0.5, 1
        data_args = stanscofi.datasets.generate_dummy_dataset(npositive, nnegative, nfeatures, mean, std)
        dataset = stanscofi.datasets.Dataset(**data_args)
        return dataset

    ## Test whether basic functions of the model work
    def test_model(self):  
        np.random.seed(123456)
        dataset = self.generate_dataset()
        #test_size = 0.3  
        #train_set, test_set, _, _ = stanscofi.training_testing.traintest_validation_split(dataset, test_size, early_stop=1, metric="euclidean", disjoint_users=False, random_state=123456, verbose=False, print_dists=False)
        #train_dataset = dataset.get_folds(train_set.astype(int))
        #test_dataset = dataset.get_folds(test_set.astype(int))
        train_dataset, test_dataset = dataset, dataset
        model = benchscofi.XXXXXX.XXXXXX()
        model.fit(train_dataset)
        scores = model.predict(test_dataset)
        #tmp = np.copy(scores[:,0]) ## !!!
        #scores[:,0] = scores[:,1].tolist() ## !!!
        #scores[:,1] = tmp.tolist() ## !!!
        #print((np.max(scores[:,0]), np.max(scores[:,1])))
        #print(test_dataset.ratings_mat.shape)
        print("\n\n"+('_'*27)+"\nMODEL "+model.name)
        model.print_scores(scores)
        predictions = model.classify(scores)
        model.print_classification(predictions)
        metrics, _ = stanscofi.validation.compute_metrics(scores, predictions, test_dataset, beta=1, verbose=False)
        print(metrics)
        print(("_"*27)+"\n\n")
        ## if it ends without any error, it is a success