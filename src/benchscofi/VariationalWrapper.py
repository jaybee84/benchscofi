#coding: utf-8

from stanscofi.models import BasicModel
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from scipy.sparse import coo_array
from sklearn.metrics import roc_auc_score, mean_squared_error, average_precision_score

## https://github.com/jilljenn/vae/tree/1d7f09af3bcaebfc5d8fa8cc18033d8bb8ca19bc
from benchscofi.implementations import VariationalInference

class VariationalWrapper(BasicModel):
    def __init__(self, params=None):
        params = params if (params is not None) else self.default_parameters()
        super(VariationalWrapper, self).__init__(params)
        self.name = "VariationalWrapper"
        self.model = None

    def default_parameters(self):
        params = {
            "LEARNING_RATE" : 1,
            "N_VARIATIONAL_SAMPLES" : 1,
            "N_EPOCHS" : 5,
            "DISPLAY_EPOCH_EVERY" : 5,
            "BATCH_SIZE" : 50,
            "EMBEDDING_SIZE" : 3,
            "N_VARIATIONAL_SAMPLES" : 1,
            "optimizer": "Adam",
            "random_state": 1234,
        }
        return params

    def preprocessing(self, dataset, is_training=True):
        N, M = len(dataset.user_list), len(dataset.item_list)
        ## Ratings between 0 and 1: -1 -> 0, 0 -> 0.5, 1 -> 1
        y = dataset.ratings.toarray()[dataset.folds.row, dataset.folds.col].ravel()
        #y[y==0] = 0.5 
        #y[y==-1] = 0.
        X = np.asarray(np.zeros((y.shape[0], N+M)), dtype=np.float64)
        for x in range(y.shape[0]):
            X[x,dataset.folds.col[x]] = 1.
            X[x,dataset.folds.row[x]+N] = 1.
        return [torch.LongTensor(X), torch.LongTensor(y), N, M] if (is_training) else [torch.LongTensor(X)]

    ## source: https://raw.githubusercontent.com/jilljenn/vae/1d7f09af3bcaebfc5d8fa8cc18033d8bb8ca19bc/vfm-torch.py
    def model_fit(self, X, y, N, M):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        torch_dataset = torch.utils.data.TensorDataset(X, y)
        nb_samples = len(y)
        train_rmse = train_auc = train_map = 0.
        losses = []
        all_preds = []
        self.BATCH_SIZE = min(self.BATCH_SIZE, nb_samples//2)
        train_iter = torch.utils.data.DataLoader(torch_dataset, batch_size=self.BATCH_SIZE) # , shuffle=True
        self.model = VariationalInference.CF(self.EMBEDDING_SIZE, self.N_VARIATIONAL_SAMPLES, N=N, M=M,
                output="class")
                #output="reg")
        self.optimizer = {
            "Adam": torch.optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE), #weight_decay=1e-4)
            "LBFGS": torch.optim.LBFGS(self.model.parameters(), lr=self.LEARNING_RATE, history_size=10, max_iter=4, line_search_fn='strong_wolfe'),
            "SGD": torch.optim.SGD(self.model.parameters(), lr=self.LEARNING_RATE),
        }[self.optimizer]
        for epoch in tqdm(range(self.N_EPOCHS)):
            losses = []
            pred = []
            truth = []
            for i, (indices, target) in enumerate(train_iter):
                # print('=' * 10, i)
                outputs, _, _, kl_term = self.model(indices)#.squeeze()
                # print(outputs)
                # print('indices', indices.shape, 'target', target.shape, outputs, 'ypred', len(y_pred), 'kl', kl_term.shape)
                # loss = loss_function(outputs, target)
                # print('kl', kl_bias.shape, kl_entity.shape)
                # print(outputs.sample()[:5], target[:5])
                loss = -outputs.log_prob(target.float()).mean() * nb_samples + kl_term
                # print('loss', loss)
                train_auc = -1

                y_pred = outputs.mean.squeeze().detach().numpy().tolist()
                losses.append(loss.item())
                pred.extend(y_pred)
                truth.extend(target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # print('preds', len(y_pred))
                # print('but target', target.shape)
                # print(len(pred), len(truth))
            # optimizer.swap_swa_sgd()

            # End of epoch
            train_auc = roc_auc_score(truth, pred)
            train_map = average_precision_score(truth, pred)

            '''print('test', outputs.sample()[:5], target[:5], loss.item())
                print('variance', torch.sqrt(1 / model.alpha))
                print('bias max abs', model.bias_params.weight.abs().max())
                print('entity max abs', model.entity_params.weight.abs().max())'''

            if epoch % self.DISPLAY_EPOCH_EVERY == 0:
                print('train pred', np.round(pred[:5], 4), truth[:5])
                print(f"Epoch {epoch}: Elbo {np.mean(losses):.4f} " +
                    (f"Minibatch train AUC {train_auc:.4f} " +
                        f"Minibatch train MAP {train_map:.4f}"))

                print('precision', self.model.alpha, 'std dev', torch.sqrt(1 / nn.functional.softplus(self.model.alpha)))
                # print('bias max abs', self.model.bias_params.weight.abs().max())
                # print('entity max abs', self.model.entity_params.weight.abs().max())

    def model_predict_proba(self, X):
        outputs, _, _, _ = self.model(X)
        y_pred = outputs.mean.squeeze().detach().numpy()
        return y_pred