#coding:utf-8

from scipy.optimize import minimize, minimize_scalar
from functools import reduce

import pandas as pd
import numpy as np

## N: total number of (item, user) pairs
## nfeatures: number of features
def generate_Censoring_dataset(pi=0.3,c=0.3,N=100,nfeatures=50,mean=0.5,std=1,exact=True,random_state=123435):
    assert nfeatures%2==0
    assert pi>0 and pi<1
    assert c>0 and c<1
    np.random.seed(random_state)
    ## Generate feature matrices for unlabeled samples
    Nsqrt = int(np.sqrt(N)+1)
    if (exact):
        NPos, NNeg = int(pi*Nsqrt), Nsqrt-int(pi*Nsqrt)
    else:
        NN = np.random.binomial(1, pi, size=Nsqrt)
        NPos, NNeg = np.sum(NN), np.sum(NN==0)
    assert NPos+NNeg==Nsqrt
    ### All user feature vectors
    users = np.random.normal(0,std,size=(nfeatures//2,Nsqrt))
    ### All positive pairs
    PosItems = np.random.normal(mean,std,size=(nfeatures//2,NPos))
    ### All negative pairs
    NegItems = np.random.normal(-mean,std,size=(nfeatures//2,NNeg))
    ### All item feature vectors
    items = np.concatenate((PosItems, NegItems), axis=1)
    ### True label matrix
    labels_mat = np.asarray(np.zeros((Nsqrt,Nsqrt)), dtype=int)
    labels_mat[:NPos,:] = 1
    labels_mat[NPos:,:] = -1
    ## Generate accessible ratings = y among positive samples with probability c
    if (exact):
        ids_ls = list(range(Nsqrt*NPos))
        np.random.shuffle(ids_ls)
        NlabPos = np.asarray(np.zeros(Nsqrt*NPos), dtype=int)
        NlabPos[ids_ls[:int(c*Nsqrt*NPos)]] = 1
    else:
        NlabPos = np.random.binomial(1, c, size=Nsqrt*NPos)
    ratings_mat = np.copy(labels_mat)
    ratings_mat[:NPos,:] *= NlabPos.reshape((NPos, Nsqrt)) ## hide some of the positive
    ratings_mat[NPos:,:] = 0 ## hide all negative
    ## Input to stanscofi
    user_list, item_list, feature_list = [list(map(str,x)) for x in [range(Nsqrt), range(Nsqrt), range(nfeatures//2)]]
    ratings_mat = pd.DataFrame(ratings_mat, columns=user_list, index=item_list).astype(int)
    labels_mat = pd.DataFrame(labels_mat, columns=user_list, index=item_list).astype(int)
    users = pd.DataFrame(users, index=feature_list, columns=user_list)
    items = pd.DataFrame(items, index=feature_list, columns=item_list)
    return {"ratings": ratings_mat, "users": users, "items": items}, labels_mat

## N: total number of datapoints
## nfeatures: number of features
## Case-Control setting
def generate_CaseControl_dataset(N=100,nfeatures=50,pi=0.3,sparsity=0.01,imbalance=0.03,mean=0.5,std=1,exact=True,random_state=123435):
    assert nfeatures%2==0
    assert pi>0 and pi<1
    assert sparsity>0 and sparsity<1
    np.random.seed(random_state)
    ## Generate feature matrices for unlabeled samples (from positive dist with probability pi)
    Nsqrt = int(np.sqrt(N))
    if (exact):
        NPos = int(pi*np.sqrt(N))
        NNeg = Nsqrt-NPos
    else:
        NIsPos = np.random.binomial(1, pi, size=Nsqrt)
        NPos, NNeg = np.sum(NIsPos), np.sum(NIsPos==0)
    ### All user feature vectors
    users = np.random.normal(0,std,size=(nfeatures//2,Nsqrt))
    ## Concatenated item feature vectors for positive and negative pairs
    PosItems = np.random.normal(mean,std,size=(nfeatures//2,NPos))
    NegItems = np.random.normal(-mean,std,size=(nfeatures//2,NNeg))
    items = np.concatenate((PosItems, NegItems), axis=1)
    ### True label matrix
    labels_mat = np.asarray(np.zeros((Nsqrt,Nsqrt)), dtype=int)
    labels_mat[:NPos,:] = 1
    labels_mat[NPos:,:] = -1
    ## Generate accessible ratings = y
    ratings_mat = np.copy(labels_mat)*0
    Ni = sparsity/(pi*(1+imbalance))
    Nip = (sparsity-pi*Ni)/(1-pi)
    NNegLab = int(NNeg*Nsqrt*Nip)
    NPosLab = int(NPos*Nsqrt*Ni)
    ids_user_ls = list(range(NPos*Nsqrt))
    np.random.shuffle(ids_user_ls)
    select_pos = np.asarray(np.zeros(NPos*Nsqrt), dtype=int)
    select_pos[ids_user_ls[:NPosLab]] = 1
    select_pos = select_pos.reshape((NPos, Nsqrt))
    ratings_mat[:NPos,:] = select_pos
    ids_user_ls = list(range(NNeg*Nsqrt))
    np.random.shuffle(ids_user_ls)
    select_neg = np.asarray(np.zeros(NNeg*Nsqrt), dtype=int)
    select_neg[ids_user_ls[:NNegLab]] = -1
    select_neg = select_neg.reshape((NNeg, Nsqrt))
    ratings_mat[NPos:,:] = select_neg
    ## Input to stanscofi
    user_list, item_list, feature_list = [list(map(str,x)) for x in [range(Nsqrt), range(Nsqrt), range(nfeatures//2)]]
    ratings_mat = pd.DataFrame(ratings_mat, columns=user_list, index=item_list).astype(int)
    labels_mat = pd.DataFrame(labels_mat, columns=user_list, index=item_list).astype(int)
    users = pd.DataFrame(users, index=feature_list, columns=user_list)
    items = pd.DataFrame(items, index=feature_list, columns=item_list)
    return {"ratings": ratings_mat, "users": users, "items": items}, labels_mat

## Charles Elkan and Keith Noto. Learning classifiers from only positive and unlabeled data. In Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining, pages 213–220, 2008.
def data_aided_estimation(scores_all, true_all, estimator_type=[1,2,3][0]): 
    assert estimator_type in [1,2,3]
    assert (scores_all>=0).all() and (scores_all<=1).all()
    assert scores_all.shape[0] == true_all.shape[0]
    sum_pos = (true_all>0).astype(int).dot(scores_all)
    size = len(scores_all)
    if (estimator_type == 1):
        est_c = sum_pos/np.sum(true_all>0)
    elif (estimator_type==2):
        est_c = sum_pos/np.sum(scores_all)
    else:
        est_c = np.max(scores_all) ## in the paper (but mean is used in pulearn)
    est_pi = sum_pos/(est_c*size)
    return est_c, est_pi

## https://arxiv.org/pdf/1306.5056.pdf
from sklearn.metrics import roc_curve as ROC
def roc_aided_estimation(scores_all, true_all, regression_type=[1,2][0]):
    assert regression_type in [1,2]
    assert scores_all.shape[0] == true_all.shape[0]
    fpr, tpr, _ = ROC(true_all, scores_all)
    #if (regression_type==1):
    #    import matplotlib.pyplot as plt
    #    plt.plot(fpr, tpr, "b-")
    #    plt.plot(fpr, fpr, "k--")
    #    plt.title("ROC curve")
    #    plt.show()
    base_fpr = np.linspace(0.001, 0.999, 101) ## alpha false positive rate
    mean_tprs = np.interp(base_fpr, fpr, tpr)
    mean_tprs[0] = 0.0
    ## Empirical (average-user) ROC curve X=base_fpr, Y=mean_tprs
    ## Fit empirical ROC curve onto regression models from C. Lloyd. Regression models for convex ROC curves. Biometrics, 56(3):862–867, September 2000.
    from scipy.special import xlogy
    def binomial_deviance(x):
        log = lambda m : xlogy(np.sign(m), m)
        import warnings
        warnings.simplefilter("ignore") #invalid value in power
        power = lambda m, p : np.power(m,p) #np.sign(m)*(np.abs(m))**p
        #def power(m,p):
        #    try:
        #        import warnings
        #        warnings.simplefilter("error")
        #        m[m<0] = 0
        #        return np.power(m,p)
        #    except:
        #        print(m)
        #        print(p)
        #        raise ValueError
        if (regression_type==1):
            gamma, Delta = x.tolist()
            def f(alpha):
                Phi = norm(loc=0.0, scale=1.0)
                Q, inv_Q = np.vectorize(Phi.cdf), np.vectorize(Phi.ppf)
                val = (1-gamma)*Q(inv_Q(alpha)+Delta)+gamma*alpha
                return val
        else:
            gamma, Delta, mu = x.tolist()
            def f(alpha):
                val = (1-gamma)*power(1+Delta*(1/power(alpha,mu)-1), -1/mu)+gamma*alpha
                return val
        return -2*np.sum( np.multiply(mean_tprs, log(f(base_fpr))) + np.multiply(1-mean_tprs, log(1-f(base_fpr))) )
    x0 = np.array([1]*(regression_type+1)) ## gamma, Delta(, mu if regression_type=2)
    res = minimize(binomial_deviance, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True, "maxiter": 1000})
    args = res.x.tolist()
    if (regression_type == 1):
        return args[0]
    return (1-args[0])*args[1]+args[0]

## https://proceedings.mlr.press/v45/Christoffel15.pdf (L1-Distance)
## https://arxiv.org/pdf/1206.4677.pdf (Pearson)
def divergence_aided_estimation(X, y, lmb=1, sigma=1., divergence_type=["L1-distance","Pearson"][0]):
    assert divergence_type in ["L1-distance", "Pearson"]
    from scipy.stats import multivariate_normal
    from scipy.spatial.distance import cdist
    pos_x = (y==1).astype(int)
    unl_x = (y<1).astype(int) # (y==0).astype(int)
    basis_mat = np.exp(-cdist(X,X,metric='euclidean')/(2*sigma**2))
    if (divergence_type=="L1-distance"):
        def approx_div(pi):
            betas = [pi/np.sum(pos_x)*basis_mat[l,pos_x].sum() if (pos_x.sum()>0) else 0 for l in range(basis_mat.shape[0])]
            betas = [betas[l]-(1/np.sum(unl_x)*basis_mat[l,unl_x].sum() if (unl_x.sum()>0) else 0) for l in range(basis_mat.shape[0])]
            return (1/lmb)*np.sum([max(0,b)*b for b in betas])-pi+1
        #import matplotlib.pyplot as plt
        #plt.figure(figsize=(2,2))
        #pi_ls = [0.1*p for p in range(0,10)]
        #plt.plot(pi_ls, [approx_div(p) for p in pi_ls], "b-")
        #plt.title("penL1(pi) curve")
        #plt.show()
        #plt.close()
    else:
        R1 = np.zeros((1,basis_mat.shape[0]))
        R3 = np.eye(basis_mat.shape[0])
        R = np.concatenate((np.column_stack((0,R1)), np.column_stack((R1.T, R3))), axis=0)
        H = np.array([1/np.sum(x)*np.array([1]+[np.sum([basis_mat[l,i] for i in range(basis_mat.shape[0]) if (x[i])]) for l in range(basis_mat.shape[0])]).T for x in [unl_x, pos_x]]).T
        basis_mat = np.concatenate((np.ones((1,basis_mat.shape[1])), basis_mat), axis=0)
        basis_mat = np.concatenate((np.ones((1,basis_mat.shape[0])).T, basis_mat), axis=1)
        G = (1/basis_mat.shape[0])*np.sum([basis_mat[l,:].reshape((1,-1)).T.dot(basis_mat[l,:].reshape((1,-1))) for l in range(basis_mat.shape[0])], axis=1)
        def approx_div(pi):
            theta = np.array([1-pi, pi]).T
            GlR = np.linalg.pinv(G+lmb*R)
            return -0.5*theta.dot(H.T).dot(GlR).dot(G).dot(GlR).dot(H.dot(theta.T))+theta.dot(H.T).dot(GlR).dot(H.dot(theta.T))-0.5
    res = minimize_scalar(approx_div, bounds=(0, 1), method='bounded')
    return res.x
