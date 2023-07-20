#coding:utf-8

from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm
from scipy.spatial.distance import cdist
from scipy.special import xlogy
from sklearn.metrics import roc_curve as ROC
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

def generate_Censoring_dataset(pi=0.3,c=0.3,N=100,nfeatures=50,mean=0.5,std=1,exact=True,random_state=1234):
    '''
    Generate a synthetic dataset under the censoring setting. Assume that $s \in \{-1,1\}$ are the true labels, $y \in \{0,1\}$ are the accessible labels (note that accessible negative samples are missing), and $v \in \mathbb{R}^d$ are the feature vectors. Samples $(v,s) \sim p(v,s)$, and then are made accessible as follows $y \sim p(\cdot \mid v, s=1)$ and $\mathbb{P}(y \/neq 0 \mid v, s=-1) = p(y=-1 \mid v, s=1) = 0$.

    Reference: Elkan, Charles, and Keith Noto. "Learning classifiers from only positive and unlabeled data." Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. 2008.

    ...

    Parameters
    ----------
    pi : float
        Prob(s=1). Comprised in ]0,1[
    c : float
        Prob(s=1 | y!=0). Comprised in ]0,1[
    N : int
        Total number of (item, user) pairs
    nfeatures : int
        Total number of features. Should be a multiple of 2
    mean : float
        Mean (resp., -mean) of the distribution of positive (resp., negative) samples
    std : float
        Standard deviation of the distributions of positive and negative samples
    exact : bool
        If set to True, the proportions in pi and c are exactly respected (e.g., pi=#positive/#total) instead of drawn under a Bernouilli law of mean pi (or c)
    random_state : int
        Random seed

    Returns
    ----------
    dataset_di : dict
        Argument to feed to function stanscofi.datasets.Dataset to create a dataset
    labels_mat : pandas DataFrame
        Corresponding matrix of true labels (i.e., without unlabeled samples)
    '''
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

def generate_CaseControl_dataset(N=100,nfeatures=50,pi=0.3,sparsity=0.01,imbalance=0.03,mean=0.5,std=1,exact=True,random_state=1234):
    '''
    Generate a synthetic dataset under the case-control setting. Assume that $s \in \{-1,1\}$ are the true labels, $y \in \{-1,0,1\}$ are the accessible labels, and $v \in \mathbb{R}^d$ are the feature vectors. Positive pairs $v \sim p_+ = p(\cdot | y=+1)$, negative pairs $v \sim p_- = p(\cdot | y=-1)$, and unlabeled pairs $v \sim p_u = \pi p_+ + (1-\pi)p_-$ (where $\pi := \mathbb{P}(s = 1) \in (0,1)$ is the class-prior probability).

    Reference: Kato, Masahiro, Takeshi Teshima, and Junya Honda. "Learning from positive and unlabeled data with a selection bias." International conference on learning representations. 2018.

    ...

    Parameters
    ----------
    N : int
        Total number of (item, user) pairs
    nfeatures : int
        Total number of features. Should be a multiple of 2
    pi : float
        Prob(s=1). Comprised in ]0,1[
    sparsity : float
        Percentage of labeled samples. Comprised in ]0,1[
    imbalance : float
        Ratio of the numbers of negative and positive labeled samples
    mean : float
        Mean (resp., -mean) of the distribution of positive (resp., negative) samples
    std : float
        Standard deviation of the distributions of positive and negative samples
    exact : bool
        If set to True, the proportions in pi and c are exactly respected (e.g., pi=#positive/#total) instead of drawn under a Bernouilli law of mean pi (or c)
    random_state : int
        Random seed

    Returns
    ----------
    dataset_di : dict
        Argument to feed to function stanscofi.datasets.Dataset to create a dataset
    labels_mat : pandas DataFrame
        Corresponding matrix of true labels (i.e., without unlabeled samples)
    '''
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

def data_aided_estimation(scores_all, true_all, estimator_type=[1,2,3][0]): 
    '''
    Class prior estimation under the case-control setting, that is, 

    Assume that $s \in \{-1,1\}$ are the true labels, $y \in \{0,1\}$ are the accessible labels (note that accessible negative samples are missing), and $v \in \mathbb{R}^d$ are the feature vectors. Samples $(v,s) \sim p(v,s)$, and then are made accessible as follows $y \sim p(\cdot \mid v, s=1)$ and $\mathbb{P}(y \/neq 0 \mid v, s=-1) = p(y=-1 \mid v, s=1) = 0$.

    Three estimators $e_1$, $e_2$ and $e_3$ of $c := \mathbb{P}(s=1 \mid y \/neq 0)$. Given a trained classifier $\widehat{\/theta}$, and a validation set $\mathcal{V} := \{ (v,y) \mid y \in \{-1,0,1\}, v \in \mathbb{R}^d \}$,
    $$ e_1 := \/frac{1}{|\{v \mid (v,+1) \in \mathcal{V}\}|}\sum_{(v,+1) \in \mathcal{V}} (f_{\widehat{\/theta}}(v))_+\;;  e_2 := \/frac{\sum_{(v',+1) \in \mathcal{V}} (f_{\widehat{\/theta}}(v'))_+}{\sum_{(v,y) \in \mathcal{V}} (f_{\widehat{\/theta}}(v))_+}\;; e_3 := \max_{(v,y) \in \mathcal{V}} (f_{\widehat{\/theta}}(v))_+\;. $$

    One can retrieve an approximation of $\pi:=\mathbb{P}(s=1)$ by using $c\pi = \mathbb{P}(y=1) \/approx \sum_{(v',+1) \in \mathcal{V}} (f_{\widehat{\/theta}}(v'))_+$
$$\hat{\pi}_i := \/frac{e_i^{-1}}{|\mathcal{V}|}\sum_{(v,+1) \in \mathcal{V}} (f_{\widehat{\/theta}}(v))_+\;.$$

    Reference: Charles Elkan and Keith Noto. Learning classifiers from only positive and unlabeled data. In Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining, pages 213–220, 2008.
    '''
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

def roc_aided_estimation(scores_all, true_all, regression_type=[1,2][0], show_plot=False, verbose=False):
    '''
    Class prior estimation under the case-control setting, that is, 

    Assume that $s \in \{-1,1\}$ are the true labels, $y \in \{-1,0,1\}$ are the accessible labels, and $v \in \mathbb{R}^d$ are the feature vectors. Positive pairs $v \sim p_+ = p(\cdot | y=+1)$, negative pairs $v \sim p_- = p(\cdot | y=-1)$, and unlabeled pairs $v \sim p_u = \pi p_+ + (1-\pi)p_-$ (where $\pi := \mathbb{P}(s = 1) \in (0,1)$ is the class-prior probability).

    [1, Theorem $4$] shows that if the supports for $p_+$ and $p_-$ are different
$$\hat{\pi} = -\lim_{\substack{\/alpha \/rightarrow 1\/\/ \/alpha < 1}}\/frac{\partial}{\partial \/alpha}\inf_{\/theta \in \Theta} \left\{  \/underbrace{\mathcal{R}_\/text{0-1}(\/theta)}_\/text{Bayes regret} \mid \mathbb{E}_{v \sim p_-}\ell_{0-1}(C_\/theta(v),-1) \leq \/alpha \/right\}(\/alpha)\;.$$

    As mentioned in [2], a possible approach to approximate $\hat{\pi}$ is to regress a specific model (given in [2]) on the points of the corresponding ROC curve, and use the fitted model to extract the slope at the right-hand side of the curve, which is $\hat{\pi}$.

    [1] Scott, Clayton, and Gilles Blanchard. "Novelty detection: Unlabeled data definitely help." Artificial intelligence and statistics. PMLR, 2009.

    [2] Sanderson, Tyler, and Clayton Scott. "Class proportion estimation with application to multiclass anomaly rejection." Artificial Intelligence and Statistics. PMLR, 2014. (arxiv:1306.5056)
    '''
    assert regression_type in [1,2]
    assert scores_all.shape[0] == true_all.shape[0]
    fpr, tpr, _ = ROC(true_all, scores_all)
    if (show_plot):
        plt.plot(fpr, tpr, "b-")
        plt.plot(fpr, fpr, "k--")
        plt.title("ROC curve")
        plt.show()
        plt.close()
    base_fpr = np.linspace(0.001, 0.999, 101) ## alpha false positive rate
    mean_tprs = np.interp(base_fpr, fpr, tpr)
    mean_tprs[0] = 0.0
    ## Empirical (average-user) ROC curve X=base_fpr, Y=mean_tprs
    ## Fit empirical ROC curve onto regression models from C. Lloyd. Regression models for convex ROC curves. Biometrics, 56(3):862–867, September 2000.
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
        dev = -2*np.sum( np.multiply(mean_tprs, log(f(base_fpr))) + np.multiply(1-mean_tprs, log(1-f(base_fpr))) )
        #print((regression_type, x, dev))
        return dev
    x0 = np.array([1]*(regression_type+1)) ## gamma, Delta(, mu if regression_type=2)
    bnds = tuple([(0, 1)]+[(None, None)]*regression_type)
    res = minimize(binomial_deviance, x0, bounds=bnds, method='nelder-mead', options={'xatol': 1e-8, 'disp': verbose, "maxiter": 1000})
    args = res.x.tolist()
    if (regression_type == 1):
        return args[0]
    return (1-args[0])*args[1]+args[0]

def divergence_aided_estimation(X, y, lmb=1, sigma=1., divergence_type=["L1-distance","Pearson"][0], show_plot=False):
    '''
    Class prior estimation under the case-control setting, that is, 

    Assume that $s \in \{-1,1\}$ are the true labels, $y \in \{-1,0,1\}$ are the accessible labels, and $v \in \mathbb{R}^d$ are the feature vectors. Positive pairs $v \sim p_+ = p(\cdot | y=+1)$, negative pairs $v \sim p_- = p(\cdot | y=-1)$, and unlabeled pairs $v \sim p_u = \pi p_+ + (1-\pi)p_-$ (where $\pi := \mathbb{P}(s = 1) \in (0,1)$ is the class-prior probability).

$\lambda$ and $\sigma$ are regularization parameters, and $p$ (resp., $u$) is the total number of positive (resp., unlabeled) samples).

    Using L1-distance penalized divergence [1] amounts to minimizing the following scalar function:

    $$\hat{\pi}_\/text{L1} := \/arg\min_{\pi \in (0,1)} \/frac{1}{\lambda}\sum_{l \leq p+u} ((\/beta_l(\pi))_+)^2-\pi+1 \/text{ and } \/beta_l(\pi) := \/frac{\pi}{u}\sum_{i \leq u} \mathcal{N}(x_l, \sigma^2 \/text{Id})(x_i)-\/frac{1}{p}\sum_{j \leq p} \mathcal{N}(x_l, \sigma^2 \/text{Id})(x_j)\;.$$

    [1] Christoffel, Marthinus, Gang Niu, and Masashi Sugiyama. "Class-prior estimation for learning from positive and unlabeled data." Asian Conference on Machine Learning. PMLR, 2016. (https://proceedings.mlr.press/v45/Christoffel15.pdf)

    Using the Pearson penalized divergence [2] amounts to minimizing the following scalar function:

    $$\hat{\pi}_\/text{Pearson} := \/arg\min_{\pi \in (0,1)} -\/frac{1}{2}\left[^{1-\pi}_{\pi}\/right] H^\/top(G + \lambda R)^{-1}G(G+\lambda R)^{-1}H\left[^{1-\pi}_{\pi}\/right]^\/top+\left[^{1-\pi}_{\pi}\/right] H^\/top (G+\lambda R)^{-1} H\left[^{1-\pi}_{\pi}\/right]^\/top-\/frac{1}{2} $$

    $\/text{ and } H := \left[\/frac{1}{u}\sum_{j \leq u}\left(\mathcal{N}(x_l, \sigma^2 \/text{Id})(x_j)\/right)_{0 \leq l \leq u+p}, \/frac{1}{p}\sum_{i \leq p}\left(\mathcal{N}(x_l, \sigma^2 \/text{Id})(x_i)\/right)_{0 \leq l \leq u+p} \/right] \in \mathbb{R}^{(u+p+1) \/times 2} \;, R := \left[^{0}_{(0)_{(u+p) \/times 1}} ,^{(0)_{1 \/times (u+p)}}_{Id_{(u+p) \/times (u+p)}}\/right] \in \mathbb{R}^{(u+p+1) \/times (u+p+1)} \;,$

    $G:=\/frac{1}{u+p} \sum_{i \leq u+p} \left(\mathcal{N}(x_l, \sigma^2 \/text{Id})(x_i)\/right)_{0 \leq l \leq u+p}^\/top\left(\mathcal{N}(x_l, \sigma^2 \/text{Id})(x_i)\/right)_{0 \leq l \leq u+p} \in \mathbb{R}^{(u+p+1) \/times (u+p+1)}$ where $\/forall x, \mathcal{N}(x_0, \sigma^2 \/text{Id})(x)=1$.

    [2] Du Plessis, Marthinus Christoffel, and Masashi Sugiyama. "Semi-supervised learning of class balance under class-prior change by distribution matching." Neural Networks 50 (2014): 110-119. (arxiv:1206.4677)
    '''
    assert divergence_type in ["L1-distance", "Pearson"]
    pos_x = (y==1).astype(int)
    unl_x = (y<1).astype(int) # (y==0).astype(int)
    basis_mat = np.exp(-cdist(X,X,metric='euclidean')/(2*sigma**2))
    if (divergence_type=="L1-distance"):
        def approx_div(pi):
            betas = [pi/np.sum(pos_x)*basis_mat[l,pos_x].sum() if (pos_x.sum()>0) else 0 for l in range(basis_mat.shape[0])]
            betas = [betas[l]-(1/np.sum(unl_x)*basis_mat[l,unl_x].sum() if (unl_x.sum()>0) else 0) for l in range(basis_mat.shape[0])]
            return (1/lmb)*np.sum([max(0,b)*b for b in betas])-pi+1
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
    if (show_plot):
        plt.figure(figsize=(2,2))
        pi_ls = [0.1*p for p in range(0,10)]
        plt.plot(pi_ls, [approx_div(p) for p in pi_ls], "b-")
        plt.title("penL1(pi) curve")
        plt.show()
        plt.close()
    res = minimize_scalar(approx_div, bounds=(0, 1), method='bounded')
    return res.x
