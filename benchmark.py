#coding:utf-8

## Import

random_state = 1232454

from stanscofi.utils import load_dataset
from stanscofi.datasets import Dataset
from stanscofi.training_testing import traintest_validation_split, print_folds, cv_training
from stanscofi.validation import compute_metrics, plot_metrics

from glob import glob
from time import time

import sys
sys.path.insert(0,"src/")

import benchscofi

## Benchmark parameters

dataset_names = ["Gottlieb"]
split_params = {
    "metric": "cityblock",
    "test_size": 0.2
}

## Algorithm parameters

algo_params = {
    "Constant": None,
}

## Benchmark

for dataset_name in dataset_names:
    dataset_di = load_dataset(dataset_name, "datasets/")
    dataset_di.setdefault("same_item_user_features", dataset_name=="TRANSCRIPT")
    dataset_di.setdefault("name", dataset_name)
    dataset = Dataset(**dataset_di)
    train_set, test_set, _,_ = traintest_validation_split(dataset, test_size=split_params["test_size"], 
        early_stop=2, metric=split_params["metric"], disjoint_users=False, verbose=False, 
        random_state=random_state, print_dists=True)
    for algo in algo_params:
        __import__("benchscofi."+algo)
        model = eval("benchscofi."+algo+"."+algo)(algo_params[algo])
        if (model.use_masked_dataset):
            train_dataset = dataset.mask_dataset(train_set)
            test_dataset = dataset.mask_dataset(test_set)
        else:
            train_dataset = dataset.get_folds(train_set)
            test_dataset = dataset.get_folds(test_set)
        model.fit(train_dataset)
        scores = model.predict(test_dataset)
        print("\n\n"+('_'*27)+"\nMODEL "+model.name)
        model.print_scores(scores)
        predictions = model.classify(scores)
        model.print_classification(predictions)
        metrics, _ = stanscofi.validation.compute_metrics(scores, predictions, test_dataset, beta=1, verbose=False)
        print(metrics)

