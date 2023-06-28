#coding: utf-8

from stanscofi.models import BasicModel, create_scores
from stanscofi.preprocessing import preprocessing_routine
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, mean_squared_error, average_precision_score

from benchscofi.implementations import VariationalInference

class VariationalWrapper(BasicModel):
    def __init__(self, params=None):
        params = params if (params is not None) else self.default_parameters()
        super(VariationalWrapper, self).__init__(params)
        self.random_state = params["random_state"]
        self.name = "VariationalWrapper"
        self.model = VariationalInference.CF(params["EMBEDDING_SIZE"], output="class")
        self.optimizer = {
            "Adam": torch.optim.Adam(self.model.parameters(), lr=params["LEARNING_RATE"]), #weight_decay=1e-4)
            "LBFGS": torch.optim.LBFGS(self.model.parameters(), lr=params["LEARNING_RATE"], history_size=10, max_iter=4, line_search_fn='strong_wolfe'),
            "SGD": torch.optim.SGD(self.model.parameters(), lr=params["LEARNING_RATE"]),
        }[params.get("optimizer", "Adam")]
        self.N_EPOCHS = params["N_EPOCHS"]
        self.DISPLAY_EPOCH_EVERY = params["DISPLAY_EPOCH_EVERY"]
        self.BATCH_SIZE = params["BATCH_SIZE"]

    def default_parameters(self):
        params = {
            "LEARNING_RATE" : 1,
            "N_VARIATIONAL_SAMPLES" : 1,
            "N_EPOCHS" : 100,
            "DISPLAY_EPOCH_EVERY" : 5,
            "BATCH_SIZE" : 100,
            "EMBEDDING_SIZE" : 3,
            "optimizer": "Adam",
        }
        params.update({"random_state": 1354, "decision_threshold": 1})
        return params

    def preprocessing(self, dataset):
        X, y = spreprocessing_routine(dataset) ## TODO
        X, y = torch.LongTensor(X), torch.LongTensor(y)
        return X, y
        
    def fit(self, train_dataset):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        X, y = self.preprocessing(train_dataset)
        torch_dataset = torch.utils.data.TensorDataset(X, y)
        nb_samples = len(y)
	train_rmse = train_auc = train_map = 0.
	losses = []
	all_preds = []
	train_iter = torch.utils.data.DataLoader(torch_dataset, batch_size=self.BATCH_SIZE) # , shuffle=True
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

    def model_predict(self, test_dataset):
        X, y = self.preprocessing(test_dataset)
        outputs, y_pred_of_last, y_pred_of_mean, _ = self.model(X)
        y_pred = outputs.mean.squeeze().detach().numpy()
        scores = create_scores(y_pred, test_dataset)
        return scores