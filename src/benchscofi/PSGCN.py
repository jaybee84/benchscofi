#coding: utf-8

## https://github.com/bbjy/PSGCN/tree/e9edea02d76e3593fe678c8cabf82dc6aaa3a65a
import torch
from stanscofi.models import BasicModel
from torch_geometric.data import DataLoader
from benchscofi.implementations import PSGCNImplementation
from scipy.sparse import csr_matrix
import numpy as np
import random
#from subprocess import call

#import calendar
#import time
#current_GMT = time.gmtime()

class PSGCN(BasicModel):
    def __init__(self, params=None):
        params = params if (params is not None) else self.default_parameters()
        super(PSGCN, self).__init__(params)
        assert self.preprocessing_str in ["Perlman_procedure", "meanimputation_standardize", "same_feature_preprocessing"]
        self.model = None
        self.scalerS, self.scalerP, self.filter = None, None, None
        self.name = "PSGCN"

    def default_parameters(self):
        params = {
            "seed": 1234,
            "hop": 2, # number of neighbor
            "reg_lambda": 0.002,
            "k": 30,
            "lr": 1e-3, # learning rat
            "latent_dim": [64, 64, 1],
            "epochs": 30, # number of epochs
            "batch_size": 128, # batch size during training
            "dropout": 0.4, # random drops neural node and edge with this prob
            "force_undirected": False, # in edge dropout, force (x, y) and (y, x) to be dropped together
            "valid_interval": 1, 
            "device": "cpu", #"cuda"
            "preprocessing_str": "meanimputation_standardize", "subset": None,
        }
        #time_stamp = calendar.timegm(current_GMT)+np.random.choice(range(int(1e8)), size=1)[0]
        #self.filefolder = "MBiRW_%s" % time_stamp 
        return params

    def preprocessing(self, dataset, is_training=True):
        #if (is_training): ## only 1's
        #    vals = dataset.ratings.toarray()
        #    vals[vals<=0] = 0
        #elif (not is_training): ## half 0's and half 1's
        #   vals = np.zeros(dataset.ratings.shape)
        #   vals[dataset.folds.row,dataset.folds.col] = 1
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        vals = dataset.ratings.toarray()
        vals[vals<=0] = 0
        split_data_dict = PSGCNImplementation.load_k_fold(vals, self.seed)
        graphs = PSGCNImplementation.extract_subgraph(split_data_dict, self, k=0, is_training=is_training)#,filefolder=filefolder)
        if (self.model is None):
            self.model = PSGCNImplementation.PSGCN(graphs, latent_dim=self.latent_dim, k=self.k, dropout=self.dropout, force_undirected=self.force_undirected)
        return [graphs] if (is_training) else [graphs, dataset.folds.data.shape[0], vals.shape, dataset.folds]

    def model_fit(self, train_graphs):
        PSGCNImplementation.train_multiple_epochs(train_graphs, train_graphs, self.model, self)
        #call("rm -rf %s/" % self.filefolder, shell=True)
        self.model.eval()

    def model_predict_proba(self, test_graphs, n, shp, fds):
        test_loader = DataLoader(test_graphs, n, shuffle=False, num_workers=1)
        for data in test_loader:  
            y_true = data.y.view(-1).cpu().detach().numpy()
            with torch.no_grad():
                y_pred = self.model(data).cpu().detach().numpy()
        assert (y_true == test_graphs.labels).all()
        outs = np.zeros(shp)  
        outs[test_graphs.links[0],test_graphs.links[1]] = y_pred
        outs = outs[fds.row, fds.col].flatten() 
        return outs