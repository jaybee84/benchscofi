#coding:utf-8

import numpy as np

## https://github.com/cjlin1/libmf/tree/4e14bc7e8bd429abd1bbec2abc7c3926aaa3fb10 
## conversion from C++ to Python
def calc_mpr_auc(scores, dataset, transpose=False): 
    n, m = dataset.ratings.T.shape if (transpose) else dataset.ratings.shape
    nnz = dataset.ratings.data.shape[0]
    y_true_all = dataset.ratings.toarray()[dataset.ratings.row,dataset.ratings.col]
    y_true_all = y_true_all.ravel() if (transpose) else y_true_all.T.ravel()
    scores_all = scores.data.ravel()
    R = np.column_stack((dataset.ratings.row,dataset.ratings.col)).tolist()
    R_dict = {(i,j): y_true_all[ii] for ii, [i,j] in enumerate(R)}
    R = sorted(R)
    sort_by_pred = lambda x, y : x[1]<y[1]
    pos_cnts = [0]*(m+1)
    for i in range(nnz):
        pos_cnts[R[i][int(not transpose)]] += 1
    for i in range(1, m+1):
        pos_cnts[i] += pos_cnts[i-1]
    total_m, total_pos, all_u_mpr, all_u_auc = 0, 0, [], []
    for i in range(m):
        if (pos_cnts[i+1]-pos_cnts[i] < 1):
            continue
        row = [((i,j),R_dict.get((i,j),0)) for j in range(n)]
        pos = 0
        index = [0]*(pos_cnts[i+1]-pos_cnts[i])
        for j in range(pos_cnts[i], pos_cnts[i+1]):
            if (y_true_all[j]<=0):
                continue
            col = R[j][1]
            #row[col].first.r = prob->R[j].r
            index[pos] = col
            pos += 1
        if ((n-pos < 1) or (pos < 1)):
            continue
        total_m += 1
        total_pos += pos
        count = 0
        for k in range(pos):
            tmp = row[count]
            row[count] = row[index[k]]
            row[index[k]] = tmp
            count += 1
        row = [row[i] for i in np.argsort([r for _,r in row])]
        row.reverse()
        u_mpr, u_auc = 0, 0
        for neg_it in range(pos, len(row)):
            if (row[pos-1][1] <= row[neg_it][1]):
                u_mpr += pos
                continue
            left, right = 0, pos-1
            while (left<right):
                mid = int((left+right)/2)
                if (row[mid][1]>row[neg_it][1]):
                    right = mid
                else:
                    left = mid+1
            u_mpr += left
            u_auc += pos-left
        all_u_mpr.append(u_mpr/(n-pos))
        all_u_auc.append(u_auc/(n-pos)/pos)
    return all_u_mpr, all_u_auc, np.sum(all_u_mpr)/total_pos, np.sum(all_u_auc)/total_m

## Using the Lin et al. formula for AUC
## https://www.csie.ntu.edu.tw/~cjlin/papers/one-class-mf/biased-mf-sdm-with-supp.pdf
def calc_auc(scores, dataset, transpose=False, verbose=False): 
    user_ids = np.unique(dataset.folds.col) if (transpose) else np.unique(dataset.folds.row)
    n_ignored = 0
    y_true_all = dataset.ratings.toarray()[dataset.folds.row,dataset.folds.col].ravel() 
    scores_all = scores.data.ravel()
    assert y_true_all.shape==scores_all.shape
    rowwise_aucs = []
    for user_id in user_ids:
        user_ids_i = np.argwhere(dataset.folds.col==user_id) if (transpose) else np.argwhere(dataset.folds.row==user_id)
        if (len(user_ids_i)==0):
            #rowwise_aucs.append(0)
            n_ignored += 1
            continue
        user_truth = y_true_all[user_ids_i].ravel()
        user_pred = scores_all[user_ids_i].ravel()
        if ((len(np.unique(user_truth))==2) and (1 in user_truth)):
            ## Full
            omegaplus, omegaminus = np.sum(user_truth==1), np.sum(user_truth<1)
            if (omegaplus*omegaminus==0):
                #rowwise_aucs.append(0)
                n_ignored += 1
                continue
            num = sum([int(user_pred[j]>user_pred[jp]) for j in np.argwhere(user_truth==1) for jp in np.argwhere(user_truth<1)])
            auc = num/(omegaplus*omegaminus)
            rowwise_aucs.append(auc)
        else:
            #rowwise_aucs.append(0)
            n_ignored += 1
            continue
    if (verbose):
        print("#ignored users = %d (%d perc.)" % (n_ignored, n_ignored*100/len(user_ids)))
    return rowwise_aucs