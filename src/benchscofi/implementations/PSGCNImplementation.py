# coding:utf-8

## concatenation of contents in https://github.com/bbjy/PSGCN/tree/e9edea02d76e3593fe678c8cabf82dc6aaa3a65a

import torch
from torch_geometric.nn import GCNConv

from torch.optim import Adam

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import math
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear, Conv1d
from torch_geometric.nn import GCNConv, global_sort_pool
from torch_geometric.utils import dropout_adj

import os
import torch
import time
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as ssp
import multiprocessing as mp

from tqdm import tqdm
from scipy import io
from torch_geometric.data import Data, InMemoryDataset, Dataset
from sklearn.model_selection import KFold

import warnings
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch

from torch_geometric.data import DataLoader
from sklearn import metrics

import torch
from torch_geometric.nn import GCNConv

def extract_subgraph(split_data_dict, args, k=0, is_training=True, filefolder="data"):
    #if args.data_name == 'Gdataset':
    #    print("Using Gdataset with 10% testing...")
    #    (
    #        adj_train, train_labels, train_u_indices, train_v_indices,
    #        test_labels, test_u_indices, test_v_indices
    #    ) = split_data_dict[k]
    #
    #elif args.data_name == 'Cdataset':
    #
    #    print("Using Cdataset with 10% testing...")
    #    (
    #        adj_train, train_labels, train_u_indices, train_v_indices,
    #        test_labels, test_u_indices, test_v_indices
    #    ) = split_data_dict[k]  # load_drug_data(path)
    #
    #else:
    #    print("Using LRSSL with 10% testing...")
    #    (
    #        adj_train, train_labels, train_u_indices, train_v_indices,
    #        test_labels, test_u_indices, test_v_indices
    #    ) = split_data_dict[k]

    (
        adj_train, train_labels, train_u_indices, train_v_indices#,
        #test_labels, test_u_indices, test_v_indices
    ) = split_data_dict[k]

    val_test_appendix = str(k) + '_kfold'
    data_combo = (filefolder+"_"+("train" if (is_training) else "test"), val_test_appendix) #(args.data_name, val_test_appendix)

    train_indices = (train_u_indices, train_v_indices)
    #test_indices = (test_u_indices, test_v_indices)

    train_file_path = '{}/{}/'.format(*data_combo)
    if (False):#is_training):
        train_graph = MyDataset(train_file_path, adj_train, train_indices, train_labels, args.hop)
    else:
        train_graph = MyDynamicDataset(train_file_path, adj_train, train_indices, train_labels, args.hop)

    #test_file_path = 'data/{}/{}/test'.format(*data_combo)
    #test_graph = MyDataset(test_file_path, adj_train, test_indices, test_labels, args.hop)
    # test_graph = MyDynamicDataset(test_file_path, adj_train, test_indices, test_labels, args.hop)

    return train_graph#, test_graph

class attention_score(torch.nn.Module):
    def __init__(self, in_channels, Conv=GCNConv):
        super(attention_score, self).__init__()
        self.in_channels = in_channels
        self.score_layer = Conv(in_channels, 1)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        score = self.score_layer(x, edge_index)

        return score


def train_multiple_epochs(train_dataset, test_dataset, model, args):
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=2)

    test_size = 1024  # load all test dataset
    test_loader = DataLoader(test_dataset, test_size, shuffle=False,
                             num_workers=2)

    model.to(args.device).reset_parameters()
    optimizer = Adam(model.parameters(), lr=args.lr)

    start_epoch = 1
    pbar = range(start_epoch, args.epochs + start_epoch)

    for epoch in pbar:
        train_loss = train(model, optimizer, train_loader, args.device)

        if epoch % args.valid_interval == 0:
            roc_auc, aupr = evaluate_metric(model, test_loader, args.device)

            print("epoch {}".format(epoch), "train_loss {0:.4f}".format(train_loss),
                  "roc_auc {0:.4f}".format(roc_auc), "aupr {0:.4f}".format(aupr))


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader, device):
    model.train()
    total_loss = 0
    pbar = loader
    for data in pbar:
        optimizer.zero_grad()
        true_label = data.to(device)
        predict = model(true_label)
        loss_function = torch.nn.BCEWithLogitsLoss()
        loss = loss_function(predict, true_label.y.view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
        torch.cuda.empty_cache()

    return total_loss / len(loader.dataset)


def evaluate_metric(model, loader, device):
    model.eval()
    pbar = loader
    roc_auc, aupr = None, None
    for data in pbar:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)

        y_true = data.y.view(-1).cpu().tolist()
        y_score = out.cpu().numpy().tolist()

        fpr, tpr, threshold = metrics.roc_curve(y_true, y_score)
        roc_auc = metrics.auc(fpr, tpr)

        aupr = metrics.average_precision_score(y_true, y_score)
        torch.cuda.empty_cache()

    return roc_auc, aupr

class SparseRowIndexer:
    def __init__(self, csr_matrix):
        data = []
        indices = []
        indptr = []

        for row_start, row_end in zip(csr_matrix.indptr[:-1], csr_matrix.indptr[1:]):
            data.append(csr_matrix.data[row_start:row_end])
            indices.append(csr_matrix.indices[row_start:row_end])
            indptr.append(row_end - row_start)  # nnz of the row

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csr_matrix.shape

    def __getitem__(self, row_selector):
        indices = np.concatenate(self.indices[row_selector])
        data = np.concatenate(self.data[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))
        shape = [indptr.shape[0] - 1, self.shape[1]]
        return ssp.csr_matrix((data, indices, indptr), shape=shape)


class SparseColIndexer:
    def __init__(self, csc_matrix):
        data = []
        indices = []
        indptr = []

        for col_start, col_end in zip(csc_matrix.indptr[:-1], csc_matrix.indptr[1:]):
            data.append(csc_matrix.data[col_start:col_end])
            indices.append(csc_matrix.indices[col_start:col_end])
            indptr.append(col_end - col_start)

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csc_matrix.shape

    def __getitem__(self, col_selector):
        indices = np.concatenate(self.indices[col_selector])
        data = np.concatenate(self.data[col_selector])
        indptr = np.append(0, np.cumsum(self.indptr[col_selector]))

        shape = [self.shape[0], indptr.shape[0] - 1]
        return ssp.csc_matrix((data, indices, indptr), shape=shape)


class MyDataset(InMemoryDataset):
    def __init__(self, root, A, links, labels, hop):
        self.Arow = SparseRowIndexer(A)
        self.Acol = SparseColIndexer(A.tocsc())
        self.links = links
        self.labels = labels
        self.hop = hop
        super(MyDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        name = 'data.pt'
        return [name]

    def process(self):
        # Extract enclosing subgraphs and save to disk
        data_list = links2subgraphs(self.Arow, self.Acol, self.links, self.labels, self.hop)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        del data_list


class MyDynamicDataset(Dataset):
    def __init__(self, root, A, links, labels, h):
        super(MyDynamicDataset, self).__init__(root)
        self.Arow = SparseRowIndexer(A)
        self.Acol = SparseColIndexer(A.tocsc())
        self.links = links
        self.labels = labels
        self.h = h

    #@abstractmethod
    def len(self):# -> int:
        return len(self.links[0])

    def get(self, idx):
        i, j = self.links[0][idx], self.links[1][idx]
        g_label = self.labels[idx]
        tmp = subgraph_extraction_labeling(
            (i, j), self.Arow, self.Acol, g_label, self.h)
        return construct_pyg_graph(*tmp[0:6])


def links2subgraphs(Arow, Acol, links, labels, hop):
    # extract enclosing subgraphs
    print('Enclosing subgraph extraction begins...')
    g_list = []
    # with tqdm(total=len(links[0])) as pbar:
    #     num = 0
    #     for i, j, g_label in zip(links[0], links[1], labels):
    #         tmp = subgraph_extraction_labeling(
    #             (i, j), Arow, Acol, g_label, hop
    #         )
    #         data = construct_pyg_graph(*tmp)
    #         g_list.append(data)
    #         pbar.update(1)
    #         num = num + tmp[-1]

    start = time.time()
    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap_async(
                subgraph_extraction_labeling,
                [
                    ((i, j), Arow, Acol, g_label)
                    for i, j, g_label in 
                    zip(links[0], links[1], labels)
                ]
            )
    remaining = results._number_left
    pbar = tqdm(total=remaining)
    while True:
        pbar.update(remaining - results._number_left)
        if results.ready(): break
        remaining = results._number_left
        time.sleep(1)
    results = results.get()
    pool.close()
    pbar.close()
    end = time.time()
    print("Time elapsed for subgraph extraction: {}s".format(end - start))
    print("Transforming to pytorch_geometric graphs...")
    g_list = []
    pbar = tqdm(total=len(results))
    while results:
        tmp = results.pop()
        g_list.append(construct_pyg_graph(*tmp[0:6]))
        pbar.update(1)
    pbar.close()
    end2 = time.time()
    print("Time elapsed for transforming to pytorch_geometric graphs: {}s".format(end2 - end))
            
    return g_list


def subgraph_extraction_labeling(ind, Arow, Acol, label=1, h=1):

    u_nodes, v_nodes = [ind[0]], [ind[1]]
    u_dist, v_dist = [0], [0]
    u_visited, v_visited = set([ind[0]]), set([ind[1]])
    u_fringe, v_fringe = set([ind[0]]), set([ind[1]])

    for dist in range(1, h+1):
        if len(u_fringe) == 0 or len(v_fringe) == 0:
            break

        v_fringe, u_fringe = neighbors(u_fringe, Arow), neighbors(v_fringe, Acol)
        u_fringe = u_fringe - u_visited
        v_fringe = v_fringe - v_visited
       
        u_visited = u_visited.union(u_fringe)
        v_visited = v_visited.union(v_fringe)

        u_nodes = u_nodes + list(u_fringe)
        v_nodes = v_nodes + list(v_fringe)
        u_dist = u_dist + [dist] * len(u_fringe)
        v_dist = v_dist + [dist] * len(v_fringe)

    subgraph = Arow[u_nodes][:, v_nodes]
    # remove link between target nodes
    subgraph[0, 0] = 0
    # prepare pyg graph constructor input
    u, v, r = ssp.find(subgraph)
    v += len(u_nodes)
    # num_nodes = len(u_nodes) + len(v_nodes)
    node_labels = [x*2 for x in u_dist] + [x*2+1 for x in v_dist]
    max_node_label = 2*8 + 1  # to construct initialize label trick matrix

    return u, v, r, node_labels, max_node_label, label


def construct_pyg_graph(u, v, r, node_labels, max_node_label, y):
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)  
    edge_index = torch.stack([torch.cat([u, v]), torch.cat([v, u])], 0)
    edge_type = torch.cat([r, r])
    x = torch.FloatTensor(one_hot(node_labels, max_node_label+1))
    y = torch.FloatTensor([y])
    data = Data(x, edge_index, edge_attr=edge_type, y=y)

    return data


def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    return set(A[list(fringe)].indices)


def one_hot(idx, length):
    idx = np.array(idx)
    x = np.zeros([len(idx), length])
    x[np.arange(len(idx)), idx] = 1.0
    return x


def PyGGraph_to_nx(data):
    edges = list(zip(data.edge_index[0, :].tolist(), data.edge_index[1, :].tolist()))
    g = nx.from_edgelist(edges)
    # in case some nodes are isolated
    g.add_nodes_from(range(len(data.x)))
    edge_types = {(u, v): data.edge_type[i].item() for i, (u, v) in enumerate(edges)}
    nx.set_edge_attributes(g, name='type', values=edge_types)
    node_types = dict(zip(range(data.num_nodes), torch.argmax(data.x, 1).tolist()))
    nx.set_node_attributes(g, name='type', values=node_types)
    g.graph['rating'] = data.y.item()
    return g


#def load_k_fold(data_name, seed):
#    root_path = os.path.dirname(os.path.abspath(__file__))
#    if data_name == 'lrssl':
#        # txt dataset
#        path = os.path.join(root_path, 'drug_data/{}'.format(data_name) + '.txt')
#        matrix = pd.read_table(path, index_col=0).values
#    elif data_name in ['Gdataset', 'Cdataset']:
#        path = os.path.join(root_path, 'drug_data/{}'.format(data_name) + '.mat')
#        # mat dataset
#        data = io.loadmat(path)
#        matrix = data['didr'].T
#    else:
#        # csv dataset
#        path = os.path.join(root_path, 'drug_data/{}'.format(data_name) + '.csv')
#        data = pd.read_csv(path, header=None)
#        matrix = data.values.T

def load_k_fold(matrix, seed):

    drug_num, disease_num = matrix.shape[0], matrix.shape[1]
    drug_id, disease_id = np.nonzero(matrix)

    num_len = int(np.ceil(len(drug_id) * 1))  # setting sparse ratio
    drug_id, disease_id = drug_id[0: num_len], disease_id[0: num_len]

    neutral_flag = 0
    labels = np.full((drug_num, disease_num), neutral_flag, dtype=np.int32)
    observed_labels = [1] * len(drug_id)
    labels[drug_id, disease_id] = np.array(observed_labels)
    labels = labels.reshape([-1])

    # number of test and validation edges
    #num_train = int(np.ceil(0.9 * len(drug_id)))
    #num_test = int(np.ceil(0.1 * len(drug_id)))
    #print("num_train {}".format(num_train),
    #      "num_test {}".format(num_test))

    #print("num_train, num_test's ratio is", 0.9, 0.1)

    # negative sampling
    neg_drug_idx, neg_disease_idx = np.where(matrix == 0)
    neg_pairs = np.array([[dr, di] for dr, di in zip(neg_drug_idx, neg_disease_idx)])
    np.random.seed(seed)
    np.random.shuffle(neg_pairs)
    # neg_pairs = neg_pairs[0:num_train + num_test - 1]
    neg_idx = np.array([dr * disease_num + di for dr, di in neg_pairs])

    # positive sampling
    pos_pairs = np.array([[dr, di] for dr, di in zip(drug_id, disease_id)])
    pos_idx = np.array([dr * disease_num + di for dr, di in pos_pairs])

    split_data_dict = {}
    count = 0
    #kfold = KFold(n_splits=k, shuffle=True, random_state=seed)
    #for train_data, test_data in kfold.split(pos_idx):
    for train_data in [pos_idx]:
        # train dataset contains positive and negative
        idx_pos_train = np.array(pos_idx)#[np.array(train_data)]

        idx_neg_train = neg_idx[0:len(idx_pos_train)]  # training dataset pos:neg = 1:1
        idx_train = np.concatenate([idx_pos_train, idx_neg_train], axis=0)

        pairs_pos_train = pos_pairs#[np.array(train_data)]
        pairs_neg_train = neg_pairs[0:len(pairs_pos_train)]
        pairs_train = np.concatenate([pairs_pos_train, pairs_neg_train], axis=0)

        # test dataset contains positive and negative
        #idx_pos_test = np.array(pos_idx)[np.array(test_data)]
        #idx_neg_test = neg_idx[len(pairs_pos_train): len(pairs_pos_train) + len(idx_pos_test) + 1]
        #idx_test = np.concatenate([idx_pos_test, idx_neg_test], axis=0)

        #pairs_pos_test = pos_pairs[np.array(test_data)]
        #pairs_neg_test = neg_pairs[len(pairs_pos_train): len(pairs_pos_train) + len(idx_pos_test) + 1]
        #pairs_test = np.concatenate([pairs_pos_test, pairs_neg_test], axis=0)

        # Internally shuffle training set
        rand_idx = list(range(len(idx_train)))
        np.random.seed(seed)
        np.random.shuffle(rand_idx)
        idx_train = idx_train[rand_idx]
        pairs_train = pairs_train[rand_idx]

        u_train_idx, v_train_idx = pairs_train.transpose()
        #u_test_idx, v_test_idx = pairs_test.transpose()

        # create labels
        train_labels = labels[idx_train]
        #test_labels = labels[idx_test]

        # make training adjacency matrix
        rating_mx_train = np.zeros(drug_num * disease_num, dtype=np.float32)
        rating_mx_train[idx_train] = labels[idx_train]
        rating_mx_train = ssp.csr_matrix(rating_mx_train.reshape(drug_num, disease_num))
        split_data_dict[count] = [rating_mx_train, train_labels, u_train_idx, v_train_idx]#, \
        #   test_labels, u_test_idx, v_test_idx]
        count += 1

    return split_data_dict



class PSGCN(torch.nn.Module):
    def __init__(self, dataset, gconv=GCNConv, latent_dim=[64, 64, 1], k=30,
                 dropout=0.3, force_undirected=False):
        super(PSGCN, self).__init__()

        self.dropout = dropout
        self.force_undirected = force_undirected
        self.score1 = attention_score(latent_dim[0])
        self.score2 = attention_score(latent_dim[1])
        self.score3 = attention_score(latent_dim[2])

        self.conv1 = gconv(dataset.num_features, latent_dim[0])
        self.conv2 = gconv(latent_dim[0], latent_dim[1])
        self.conv3 = gconv(latent_dim[1], latent_dim[2])

        if k < 1:  # transform percentile to number
            node_nums = sorted([g.num_nodes for g in dataset])
            k = node_nums[int(math.ceil(k * len(node_nums)))-1]
            k = max(10, k)  # no smaller than 10

        self.k = int(k)
        self.dropout = dropout
        conv1d_channels = [16, 32]
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws = [self.total_latent_dim, 5]
        self.conv1d_params1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)

        self.conv1d_params2 = Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)
        dense_dim = int((k - 2) / 2 + 1)
        if (dense_dim<conv1d_kws[1]):
            dense_dim = conv1d_kws[1]
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(self.dense_dim, 128)
        self.lin2 = Linear(128, 1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv1d_params1.reset_parameters()
        self.conv1d_params2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # delete edge_attribute
        edge_index, edge_type = dropout_adj(
            edge_index, p=self.dropout,
            force_undirected=self.force_undirected, num_nodes=len(x),
            training=self.training
        )

        x = torch.relu(self.conv1(x, edge_index))
        attention_score1 = self.score1(x, edge_index)
        x1 = torch.mul(attention_score1, x)

        x = torch.relu(self.conv2(x, edge_index))
        attention_score2 = self.score2(x, edge_index)
        x2 = torch.mul(attention_score2, x)

        x = torch.relu((self.conv3(x, edge_index)))
        attention_score3 = self.score3(x, edge_index)
        x3 = torch.mul(attention_score3, x)

        X = [x1, x2, x3]
        concat_states = torch.cat(X, 1)

        x = global_sort_pool(concat_states, batch, self.k)
        x = x.unsqueeze(1)
        x = F.relu(self.conv1d_params1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv1d_params2(x))
        x = x.view(len(x), -1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return x[:, 0]

class attention_score(torch.nn.Module):
    def __init__(self, in_channels, Conv=GCNConv):
        super(attention_score, self).__init__()
        self.in_channels = in_channels
        self.score_layer = Conv(in_channels, 1)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        score = self.score_layer(x, edge_index)

        return score