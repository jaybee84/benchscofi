#coding:utf-8

import stanscofi.preprocessing
import numpy as np

def create_scores(preds, dataset):
    assert str(type(preds)) in ["<class 'float'>", "<class 'int'>"] or len(preds.shape)==1 or preds.shape[1]==1
    ids = np.argwhere(np.ones(dataset.ratings_mat.shape))
    assert str(type(preds)) in ["<class 'float'>", "<class 'int'>"] or preds.shape[0]==ids.shape[0]
    scores = np.zeros((ids.shape[0], 3))
    scores[:,0] = ids[:,1] 
    scores[:,1] = ids[:,0] 
    scores[:,2] = np.ravel(preds)
    return scores

def preprocessing_routine(dataset, preprocessing_str, subset_=None, filter_=None, scalerS=None, scalerP=None, inf=2, njobs=1):
    assert preprocessing_str in ["Perlman_procedure","meanimputation_standardize","same_feature_preprocessing"]
    if (preprocessing_str == "Perlman_procedure"):
        X, y = eval("stanscofi.preprocessing."+preprocessing_str)(dataset, njobs=njobs, sep_feature="-", missing=-666, verbose=False)
        scalerS, scalerP = None, None
    if (preprocessing_str == "meanimputation_standardize"):
        X, y, scalerS, scalerP = eval("stanscofi.preprocessing."+preprocessing_str)(dataset, subset=subset_, scalerS=scalerS, scalerP=scalerP, inf=inf, verbose=False)
    if (preprocessing_str == "same_feature_preprocessing"):
        X, y = eval("stanscofi.preprocessing."+preprocessing_str)(dataset)
        scalerS, scalerP = None, None
    if (preprocessing_str != "meanimputation_standardize"):
        if ((subset_ is not None) or (filter_ is not None)):
            if ((subset_ is not None) and (filter_ is None)):
                with np.errstate(over="ignore"):
                    x_vars = [np.nanvar(X[:,i]) if (np.sum(~np.isnan(X[:,i]))>0) else 0 for i in range(X.shape[1])]
                    x_vars = [x if (not np.isnan(x) and not np.isinf(x)) else 0 for x in x_vars]
                    x_ids_vars = np.argsort(x_vars).tolist()
                    features = x_ids_vars[-subset_:]
                    filter_ = features
            X = X[:,filter_]
    return X, y, scalerS, scalerP, filter_