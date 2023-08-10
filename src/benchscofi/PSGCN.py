#coding: utf-8

## https://github.com/bbjy/PSGCN/tree/e9edea02d76e3593fe678c8cabf82dc6aaa3a65a
from stanscofi.models import BasicModel
from torch_geometric.data import DataLoader
from benchscofi.implementations import PSGCNImplementation
import numpy as np

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
            "epochs": 2,#30, # number of epochs
            "batch_size": 128, # batch size during training
            "dropout": 0.4, # random drops neural node and edge with this prob
            "force_undirected": False, # in edge dropout, force (x, y) and (y, x) to be dropped together
            "valid_interval": 1, 
            "device": "cpu", #"cuda"
            "preprocessing_str": "meanimputation_standardize", "subset": None,
        }
        return params

    def preprocessing(self, dataset, is_training=True):
        split_data_dict = PSGCNImplementation.load_k_fold(dataset.ratings.toarray(), self.seed)
        graphs = PSGCNImplementation.extract_subgraph(split_data_dict, self, k=0)
        if (is_training or self.model is None):
            self.model = PSGCNImplementation.PSGCN(graphs, latent_dim=self.latent_dim, k=self.k, dropout=self.dropout,
                force_undirected=self.force_undirected)
        return [graphs] if (is_training) else [graphs, dataset.folds.data.shape[0]]

    def model_fit(self, train_graphs):
        PSGCNImplementation.train_multiple_epochs(train_graphs, train_graphs, self.model, self)
        self.model.eval()

    def model_predict_proba(self, test_graphs, n):
        test_loader = DataLoader(test_graphs, n, shuffle=False, num_workers=2)
        outs = []
        current_n = 0
        for data in test_loader:       
            y_true = data.y.view(-1).cpu().detach().numpy()
            outs.append(self.model(data).cpu().detach().numpy())
            current_n += outs[-1].shape[0]
            #if (current_n>=n):
            break
        outs = np.concatenate(tuple(outs), axis=0)
        return outs