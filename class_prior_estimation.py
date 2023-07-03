#coding: utf-8

####################

from importlib import reload
import stanscofi.datasets
import stanscofi.utils
import numpy as np

import sys
sys.path.insert(0,"src/")

import benchscofi
import benchscofi.utils
from benchscofi.utils import prior_estimation

rseed=12345

####################

if (True):

	N=500
	true_pi=0.3

	npositive, nnegative, nfeatures, mean, std = int(true_pi*N), N-int(true_pi*N), 100, 1, 1
	data_args = stanscofi.datasets.generate_dummy_dataset(npositive,nnegative,nfeatures,mean,std,random_state=rseed)
	dataset = stanscofi.datasets.Dataset(**data_args)

	print("True generated pi="+str(true_pi))

	estimated_pi1=np.sum(dataset.ratings[:,2]>0)/dataset.ratings.shape[0]
	estimated_pi2=np.sum(dataset.ratings_mat>0)/np.prod(dataset.ratings_mat.shape)

	print("Estimated pi="+str(estimated_pi1)+" (#positive/#known)")
	print("Estimated pi="+str(estimated_pi2)+" (#positive/#total)")

####################

else:

	dataset_name="Gottlieb"

	data_args = stanscofi.utils.load_dataset(dataset_name, "datasets/")
	dataset = stanscofi.datasets.Dataset(**data_args)

	estimated_pi1=np.sum(dataset.ratings[:,2]>0)/dataset.ratings.shape[0]
	estimated_pi2=np.sum(dataset.ratings_mat>0)/np.prod(dataset.ratings_mat.shape)

	print("Estimated pi="+str(estimated_pi1)+" (#positive/#known)")
	print("Estimated pi="+str(estimated_pi2)+" (#positive/#total)")

####################

import benchscofi.SimpleBinaryClassifier
from stanscofi.training_testing import traintest_validation_split
from stanscofi.validation import compute_metrics, plot_metrics

test_size=0.2
metric="cityblock"

train_set, test_set, _,_ = traintest_validation_split(dataset, test_size=test_size, 
         early_stop=2, metric=metric, disjoint_users=False, verbose=False, 
         random_state=rseed, print_dists=True)

train_dataset = dataset.get_folds(train_set)
test_dataset = dataset.get_folds(test_set)

## Model parameters
params = {"decision_threshold": 0.5, "preprocessing_str": "meanimputation_standardize", 
         "layers_dims": [16,32,64], "subset": None,
         "steps_per_epoch": 10, "epochs": 50, "random_state": rseed}

model = benchscofi.SimpleBinaryClassifier.SimpleBinaryClassifier(params)
model.fit(train_dataset)

## Predict the model on the testing dataset
scores = model.predict(test_dataset)
predictions = model.classify(scores)

## Validate the model on the testing dataset
metrics, _ = compute_metrics(scores, predictions, test_dataset, ignore_zeroes=False, verbose=True)
print(metrics)

####################

#from benchscofi.utils import prior_estimation
#reload(prior_estimation)

#e1, e2, e3 = [prior_estimation.data_aided_estimation(model, dataset, estimator_type=i) for i in [1,2,3]]
#e1, e2, e3

#pi_star = prior_estimation.roc_aided_estimation(model, dataset, ignore_zeroes=False, regression_type=[1,2][0])
#pi_star

#pi_hat = prior_estimation.divergence_aided_estimation(dataset, "meanimputation_standardize", lmb=1., sigma=1., 
#                                                      divergence_type=["L1-distance","Pearson"][0])
#pi_hat 