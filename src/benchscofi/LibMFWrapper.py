#coding: utf-8

from stanscofi.models import BasicModel
from stanscofi.training_testing import random_simple_split

import os
from subprocess import Popen, call
import numpy as np

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.metrics import ndcg_score, roc_auc_score
from collections import Counter
from tqdm import tqdm
import sys

## https://github.com/jilljenn/libmf/tree/9f0d6af350ba342ee52bf28111cbe30335f3baab
def execute_ndcg(trainfile, testfile, outfiles):
    df_train = pd.read_csv(trainfile, sep=' ', names=('user', 'item', 'v'))
    df = pd.read_csv(testfile, sep=' ', names=('user', 'item', 'v'))
    df_test = df
    df['user_id'] = np.unique(df['user'], return_inverse=True)[1]
    df['item_id'] = np.unique(df['item'], return_inverse=True)[1]
    encode_user = dict(df[['user', 'user_id']].drop_duplicates().to_numpy())
    encode_item = dict(df[['item', 'item_id']].drop_duplicates().to_numpy())
    df_train['user_id'] = df_train['user'].map(encode_user)
    df_train['item_id'] = df_train['item'].map(encode_item)

    n_users = df['user'].nunique()
    n_items = df['item'].nunique()
    n_entries, _ = df.shape
    #print(n_users, df['user_id'].max())
    #print(n_items, df['item_id'].max())
    truth = csr_matrix((df['v'] * n_entries, (df['user_id'], df['item_id'])), shape=(n_users, n_items))

    for filename in outfiles:
        out = pd.read_csv(filename, names=('pred',))
        #print(filename, out.shape, df['user_id'].shape)
        sparse = csr_matrix((out['pred'], (df['user_id'], df['item_id'])), shape=(n_users, n_items))

        #print('global AUC', roc_auc_score(df['v'], out['pred']))

        df['pred'] = out['pred']

        ndcg_values = []
        ndcg10_values = []
        auc_values = []
        better_auc_values = []
        normalized_row_auc_values = []
        for user_id in tqdm(df['user_id'].unique()):
            this_user = df_test.query("user_id == @user_id")
            #print(this_user)
            this_user_pos = this_user.query("v == 1")
            nb_pos = len(this_user_pos)
            this_user_neg = this_user.query("v == 0")
            nb_neg = len(this_user_neg)
            if (nb_pos*nb_neg==0):
                ndcg_values.append(0)
                ndcg10_values.append(0)
                auc_values.append(0)
                better_auc_values.append(0)
                normalized_row_auc_values.append(0)
                continue ## ignore
            this_user_pairs = this_user_pos.merge(this_user_neg, how='cross')
            #print(this_user_pairs.head())
            okay_pairs = this_user_pairs.apply(lambda row: row['pred_x'] > row['pred_y'], axis=1)
            #print(okay_pairs)
            normalized_row_auc_values.append(sum(okay_pairs) / (nb_pos * nb_neg))
            # print(this_user_pairs.shape)
            # print(normalized_row_auc_values)

            # nb_neg = this_user['v'].value_counts().loc[0]
            # nb_pos = this_user['v'].value_counts().loc[1]
            better_auc_values.append(roc_auc_score([max(0,s) for s in this_user['v']], [max(0,s) for s in this_user['pred']]))
            # break

            #print(user_id, '=', nb_pos, nb_neg, nb_pos + nb_neg, sum(okay_pairs) / (nb_pos * nb_neg), better_auc_values[-1])

            test_set = list(set(range(n_items)) - set(df_train.query("user_id == @user_id")['item_id'].tolist()))
            # print(n_items - len(test_set))
            user_truth = truth[user_id, test_set].toarray()
            # print(Counter(user_truth.reshape(-1).tolist()))
            user_pred = sparse[user_id, test_set].toarray()
            user_truth[user_truth<0] = 0
            user_pred[user_pred<0] = 0
            ndcg_values.append(ndcg_score(user_truth, user_pred))
            ndcg10_values.append(ndcg_score(user_truth, user_pred, k=10))

            try:
                auc_values.append(roc_auc_score(user_truth.reshape(-1),
                                        user_pred.reshape(-1)))
            except ValueError as e:
                assert e.args[0]=="Only one class present in y_true. ROC AUC score is not defined in that case." ## ignore users with a single class
                #print(e) 
                #print(df.query("user_id == @user_id").shape)
                #print(df.query("user_id == @user_id and v == 1"))
                #print(df_train.query("user_id == @user_id"))
                #print(user_id, len(test_set), Counter(user_truth.tolist()[0]))
                ndcg_values.append(0)
                ndcg10_values.append(0)
                auc_values.append(0)
                better_auc_values.append(0)
                normalized_row_auc_values.append(0)
                continue ## ignore
                #break
    print(filename)
    print('ndcg =', np.mean(ndcg_values))
    print('ndcg@10 =', np.mean(ndcg10_values))
    print('row auc =', np.mean(auc_values))
    print('row-oriented auc =', np.mean(better_auc_values))
    print('normalized row-oriented auc =', np.mean(normalized_row_auc_values))
    return np.mean(ndcg_values), np.mean(ndcg10_values), np.mean(auc_values), np.mean(better_auc_values), np.mean(normalized_row_auc_values)

class LibMFWrapper(BasicModel):
    def __init__(self, params=None):
        params = params if (params is not None) else self.default_parameters()
        super(LibMFWrapper, self).__init__(params)
        self.libmf_folder = "libmf/"
        self.params = params
        if (not os.path.exists(self.libmf_folder)):
            Popen("git clone -q https://github.com/jilljenn/libmf.git".split(" "))
        if (not os.path.exists(self.libmf_folder+"mf-train") or not os.path.exists(self.libmf_folder+"mf-predict")):
            call(("cd "+self.libmf_folder+" && make"), shell=True)
        self.name = "LibMFWrapper"

    def default_parameters(self):
        params = {
            'f': 12, # loss function: squared error (L2-norm)
            'l2': 0.01, # L2-regularization parameter
            'k': 32, # number of dimensions
            "a": 0.001, # coefficient of negative entries' loss
            "s": 10, # number of threads
            "c": 0.0001, # value of negative entries
        }
        return params

    def preprocessing(self, dataset, is_training=True):
        if (not is_training): ## positive and negative (0, -1) pairs
            rats = ( dataset.ratings.toarray()[dataset.folds.toarray()>0] ).ravel()
            mat = np.column_stack((dataset.folds.row, dataset.folds.col, rats))
            #mat = mat[mat[:,-1]>0]
            return [mat, rats]
        folds, _ = random_simple_split(dataset, 0.1, metric="euclidean")
        mat_lst = []
        for fds in folds: ## only positive pairs
            sb_dt = dataset.subset(fds)
            rats = ( sb_dt.ratings.toarray()[sb_dt.folds.toarray()>0] ).ravel()
            mat = np.column_stack((sb_dt.folds.row, sb_dt.folds.col, rats))
            mat = mat[mat[:,-1]>0]
            mat_lst.append(mat)
        return mat_lst

    def model_fit(self, mat_tr, mat_te, rm=True):
        np.savetxt(self.libmf_folder+"mat.tr.txt", mat_tr, fmt='%d')
        np.savetxt(self.libmf_folder+"mat.te.txt", mat_te, fmt='%d')
        conv_args = [(x,self.params[x],"d" if (" 'int'>" in str(type(self.params[x]))) else "f") for x in self.params]
        cmd = self.libmf_folder+"mf-train "+" ".join(["-"+a+" "+(("%"+f) % v) for a,v,f in conv_args])
        cmd += " -p %s %s %s" % (self.libmf_folder+"mat.te.txt", self.libmf_folder+"mat.tr.txt", self.libmf_folder+"ocmf_model.txt")
        process = Popen(cmd.split(" "))
        process.wait()
        with open(self.libmf_folder+"ocmf_model.txt", "r") as f:
            self.model = f.read()
        if (rm):
            process = Popen(("rm -f "+self.libmf_folder+"mat.*.txt "+self.libmf_folder+"ocmf_model.txt").split(" "))
            process.wait()

    def model_predict_proba(self, mat, rats, ev=12, rm=True):
        np.savetxt(self.libmf_folder+"mat.txt", mat, fmt='%d')
        with open(self.libmf_folder+"ocmf_model.txt", "w") as f:
            f.write(self.model)
        cmd = self.libmf_folder+"mf-predict -e %d %s %s %s" % (ev, self.libmf_folder+"mat.txt", 
            self.libmf_folder+"ocmf_model.txt", self.libmf_folder+"ocmf_output.txt"
        )
        process = Popen(cmd.split(" "))
        process.wait()
        #preds = np.zeros(rats.shape[0])
        #preds[rats==1] = np.loadtxt(self.libmf_folder+"ocmf_output.txt")
        preds = np.loadtxt(self.libmf_folder+"ocmf_output.txt")
        if (rm):
            process = Popen(("rm -f "+self.libmf_folder+"mat.txt "+self.libmf_folder+"ocmf_*.txt").split(" "))
            process.wait()
        return preds