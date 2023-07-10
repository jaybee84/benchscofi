#coding:utf-8

from stanscofi.validation import compute_metrics
import stanscofi.preprocessing
from scipy.stats import norm, multivariate_normal
from scipy.optimize import minimize, minimize_scalar
from functools import reduce

import pandas as pd
import numpy as np

## N: number of unlabeled points
## nfeatures: number of features
## censoring setting
def generate_Censoring_dataset(N=100,nfeatures=50,pi=0.3,mean=0.5,std=1,exact=True,random_state=123435):
    assert nfeatures%2==0
    assert pi>0 and pi<1
    np.random.seed(random_state)
    ## Generate feature matrices for unlabeled samples
    if (exact):
        NunlPos, NunlNeg = int(pi*N), N-int(pi*N)
    else:
        Nunl = np.random.binomial(1, np.sqrt(pi), size=N)
        NunlPos, NunlNeg = np.sum(Nunl), np.sum(Nunl==0) 
    UnlPosItems = np.random.normal(mean,std,size=(nfeatures,NunlPos))
    UnlNegItems = np.random.normal(-mean,std,size=(nfeatures,NunlNeg))
    users = np.concatenate((UnlPosItems[:nfeatures//2,:], UnlNegItems[:nfeatures//2,:]), axis=1)
    items = np.concatenate((UnlPosItems[nfeatures//2:,:], UnlNegItems[nfeatures//2:,:]), axis=1)
    ## Generate accessible ratings = y
    ratingsP = np.concatenate((np.zeros((NunlPos, NunlPos)), np.ones((NunlPos, NunlNeg))), axis=1)
    ratingsN = np.concatenate((-np.ones((NunlNeg, NunlPos)), np.zeros((NunlNeg, NunlNeg))), axis=1)
    ratings = np.concatenate((ratingsP, ratingsN), axis=0)
    ## Generate true labels = s
    labelsP = np.concatenate((2*np.ones((NunlPos, NunlPos)), np.ones((NunlPos, NunlNeg))), axis=1)
    labelsN = np.concatenate((-2*np.ones((NunlNeg, NunlPos)), -np.ones((NunlNeg, NunlNeg))), axis=1)
    labels = np.concatenate((labelsP, labelsN), axis=0)
    ## Input to stanscofi
    ratings_mat = pd.DataFrame(ratings, columns=range(ratings.shape[1]), index=range(ratings.shape[0])).astype(int)
    labels_mat = pd.DataFrame(labels, columns=range(labels.shape[1]), index=range(labels.shape[0])).astype(int)
    users = pd.DataFrame(users, index=range(nfeatures//2), columns=range(users.shape[1]))
    items = pd.DataFrame(items, index=range(nfeatures//2), columns=range(items.shape[1]))
    return {"ratings_mat": ratings_mat, "users": users, "items": items}, labels_mat

## N: number of unlabeled points
## nfeatures: number of features
## Case-Control setting
def generate_CaseControl_dataset(N=100,nfeatures=50,pi=0.3,mean=0.5,std=1,exact=True,random_state=123435):
    assert nfeatures%2==0
    assert pi>0 and pi<1
    np.random.seed(random_state)
    ## Generate feature matrices for unlabeled samples
    if (exact):
        NunlPos, NunlNeg = int(pi*N), N-int(pi*N)
    else:
        Nunl = np.random.binomial(1, np.sqrt(pi), size=N)
        NunlPos, NunlNeg = np.sum(Nunl), np.sum(Nunl==0) 
    UnlPosItems = np.random.normal(mean,std,size=(nfeatures,NunlPos))
    UnlNegItems = np.random.normal(-mean,std,size=(nfeatures,NunlNeg))
    users = np.concatenate((UnlPosItems[:nfeatures//2,:], UnlNegItems[:nfeatures//2,:]), axis=1)
    items = np.concatenate((UnlPosItems[nfeatures//2:,:], UnlNegItems[nfeatures//2:,:]), axis=1)
    ## Generate accessible ratings = y
    ratingsP = np.concatenate((np.zeros((NunlPos, NunlPos)), np.ones((NunlPos, NunlNeg))), axis=1)
    ratingsN = np.concatenate((-np.ones((NunlNeg, NunlPos)), np.zeros((NunlNeg, NunlNeg))), axis=1)
    ratings = np.concatenate((ratingsP, ratingsN), axis=0)
    ## Generate true labels = s
    labelsP = np.concatenate((2*np.ones((NunlPos, NunlPos)), np.ones((NunlPos, NunlNeg))), axis=1)
    labelsN = np.concatenate((-2*np.ones((NunlNeg, NunlPos)), -np.ones((NunlNeg, NunlNeg))), axis=1)
    labels = np.concatenate((labelsP, labelsN), axis=0)
    ## Input to stanscofi
    ratings_mat = pd.DataFrame(ratings, columns=range(ratings.shape[1]), index=range(ratings.shape[0])).astype(int)
    labels_mat = pd.DataFrame(labels, columns=range(labels.shape[1]), index=range(labels.shape[0])).astype(int)
    users = pd.DataFrame(users, index=range(nfeatures//2), columns=range(users.shape[1]))
    items = pd.DataFrame(items, index=range(nfeatures//2), columns=range(items.shape[1]))
    return {"ratings_mat": ratings_mat, "users": users, "items": items}, labels_mat

## Charles Elkan and Keith Noto. Learning classifiers from only positive and unlabeled data. In Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining, pages 213–220, 2008.
def data_aided_estimation(model, val_dataset, estimator_type=[1,2,3][0]):
    assert estimator_type in [1,2,3]
    scores = model.predict(val_dataset)
    scores_all = scores[:,2].flatten()
    ## clipped in [0,1]
    scores_all -= np.min(scores_all)
    scores_all /= np.max(scores_all)
    true_all = np.array([val_dataset.ratings_mat[j,i] for i,j in scores[:,:2].astype(int).tolist()])
    if (estimator_type < 3):
        sum_pos = (true_all>0).astype(int).dot(scores_all)
        if (estimator_type == 1):
            est_pi = sum_pos/np.sum(true_all>0)
            return est_pi ## remove
        est_pi = np.sum(scores_all)/scores.shape[0] ## pi, not c
        return est_pi
    est_pi = np.max(scores_all)
    return est_pi ## remove

## https://arxiv.org/pdf/1306.5056.pdf
def roc_aided_estimation(model, val_dataset, ignore_zeroes=False, regression_type=[1,2][0]):
    assert regression_type in [1,2]
    def reg_type1(x):
        gamma, Delta = x.tolist()
        Phi = norm(loc=0.0, scale=1.0)
        Q, inv_Q = np.vectorize(Phi.cdf), np.vectorize(Phi.ppf)
        return lambda alpha : (1-gamma)*Q(inv_Q(alpha)+Delta)+gamma*alpha
    def reg_type2(x):
        gamma, Delta, mu = x.tolist()
        return lambda alpha : (1-gamma)*np.power(1+Delta*(1/np.power(alpha,mu)-1), -1/mu)+gamma*alpha
    scores = model.predict(val_dataset)
    predictions = model.classify(scores)
    _, plot_args = compute_metrics(scores, predictions, val_dataset, beta=1, ignore_zeroes=ignore_zeroes, verbose=False)
    assert len(plot_args["aucs"])>0
    ## Empirical (average-user) ROC curve X=base_fpr, Y=mean_tprs
    base_fprs = np.linspace(0, 1, 101) ## alpha false positive rate
    mean_tprs = plot_args["tprs"].mean(axis=0) ## corresponding detection rate p(alpha)
    ## Fit empirical ROC curve onto regression models from C. Lloyd. Regression models for convex ROC curves. Biometrics, 56(3):862–867, September 2000.
    def binomial_deviance(x):
        f = (reg_type1 if (regression_type==1) else reg_type2)(x)
        return -2*np.sum( np.multiply(mean_tprs, np.log(f(base_fprs))) + np.multiply(1-mean_tprs, np.log(1-f(base_fprs))) )
    x0 = np.array([1]*(regression_type+1)) ## gamma, Delta(, mu if regression_type=2)
    res = minimize(binomial_deviance, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
    args = res.x.tolist()
    if (regression_type == 1):
        return args[0]
    return (1-args[0])*args[1]+args[0]

## https://proceedings.mlr.press/v45/Christoffel15.pdf (L1-Distance)
## https://arxiv.org/pdf/1206.4677.pdf (Pearson)
def divergence_aided_estimation(val_dataset, preprocessing_str, lmb=1., sigma=1., divergence_type=["L1-distance","Pearson"][0]):
    assert divergence_type in ["L1-distance", "Pearson"]
    res = eval("stanscofi.preprocessing."+preprocessing_str)(val_dataset)
    X, y = res[0], res[1]
    pos_x = (y==1).astype(int)
    unl_x = (y<1).astype(int) # (y==0).astype(int)
    #basis = [multivariate_normal(mean=X[l,:], cov=sigma**2).pdf for l in range(X.shape[0])]
    basis = [lambda x : 1]+[lambda x: np.exp(-np.linalg.norm(x-X[l,:],2)**2/(2*sigma**2)) for l in range(X.shape[0])]
    if (divergence_type=="L1-distance"):
        def approx_div(pi):
            def beta_l(l):
                b_l = pi/np.sum(unl_x)*np.sum([basis[l](X[i,:]) for i in range(X.shape[0]) if (unl_x[i]==1)]) ## shape 1x1
                b_l -= 1/np.sum(pos_x)*np.sum([basis[l](X[i,:]) for i in range(X.shape[0]) if (pos_x[i]==1)]) ## shape 1x1
                #b_l = np.sum([(pi/np.sum(unl_x) if (unl_x[i]==1) else -1/np.sum(pos_x))*basis[l](X[i,:]) for i in range(X.shape[0])])
                return b_l
            betas = np.array([beta_l(l) for l in range(len(basis))]) ## shape |basis|x1
            betas_ = np.concatenate((betas.reshape((betas.shape[0],1)), np.zeros((betas.shape[0],1))), axis=1).max(axis=1) ## shape |basis|x1
            return 1/lmb*betas_.dot(betas)-pi+1 ## shape 1x1
    else:
        def approx_div(pi):
            theta = np.array([pi, 1-pi]) ## shape 2x1
            phi = lambda x : np.array([b(x) for b in basis]) ## shape |basis|x1
            H = 1/np.sum(pos_x)*np.sum([phi(X[i,:]) for i in range(X.shape[0]) if (pos_x[i]==1)], axis=1) ## shape (d-1)x1 (X of shape (|basis|-1)xd)
            R = np.concatenate((np.eye(len(basis)), np.zeros((1,len(basis)))), axis=0) ## shape (|basis|+1)x(|basis|+1)
            R = np.concatenate((np.zeros((len(basis)+1,1)), R), axis=1) 
            G = 1/np.sum(unl_x)*reduce(lambda x,y : x+y, [phi(X[i,:].T).dot(phi(X[i,:])) for i in range(X.shape[0]) if (unl_x[i]==1)]) ## shape dxd
            return -0.5*theta.T.dot(H.T).dot(np.linalg.pinv(G+lmb*R)).dot(G).dot(np.linalg.pinv(G+lmb*R)).dot(H.dot(theta))+theta.T.dot(H.T).dot(np.linalg.pinv(G+lmb*R)).dot(H.dot(theta))-0.5 ## H = shape = theta
    #res = minimize_scalar(approx_div, bounds=(0, 1), method='bounded')
    #return res.x
    ## Grid search
    interval = [0.1*x for x in range(10)]
    values = [approx_div(pi) for pi in interval]
    return interval[np.argmin(values)]
