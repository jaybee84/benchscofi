#coding: utf-8

from stanscofi.models import BasicModel, create_overscores
from fastai.collab import CollabDataLoaders, collab_learner
import numpy as np
import pandas as pd

#https://towardsdatascience.com/collaborative-filtering-using-fastai-a2ec5a2a4049
#https://docs.fast.ai/tutorial.collab.html#Interpretation
class FastaiCollabWrapper(BasicModel):
    def __init__(self, params=None):
        params = params if (params is not None) else self.default_parameters()
        name = "FastaiCollabWrapper"
        super(FastaiCollabWrapper, self).__init__(params)
        self.y_range = [-1.0,1.0]
        self.n_factors = params["n_factors"]
        self.weight_decay = params["weight_decay"]
        self.learning_rate = params["learning_rate"]
        self.random_state = params["random_state"]
        self.n_iterations = params["n_iterations"]
        self.decision_threshold = params["decision_threshold"]
        self.name = "FastaiCollabWrapper"
        self.use_masked_dataset = True

    def default_parameters(self):
        params = {
            "n_iterations": 3, 
            "n_factors" :20,
            "weight_decay" : 0.1,
            "learning_rate" : 5e-3,
        }
        params.update({"random_state": 1354, "decision_threshold": 0.5})
        return params

    def preprocessing(self, dataset):
        df = pd.DataFrame(dataset.ratings, index=range(dataset.ratings.shape[0]), columns=["disease","drug","rating"]).astype(int)
        return df
    
    def fit(self, train_dataset):
        np.random.seed(self.random_state)
        df = self.preprocessing(train_dataset)
        data = CollabDataLoaders.from_df(df, user_name='disease', item_name="drug", rating_name="rating", bs=64, seed=self.random_state, valid_pct=0.05)
        self.model = collab_learner(data, n_factors=self.n_factors, use_nn=True, y_range=self.y_range, emb_szs=None)
        self.model.fit_one_cycle(self.n_iterations, self.learning_rate, wd=self.weight_decay)
    
    def model_predict(self, test_dataset):
        ## https://docs.fast.ai/tutorial.tabular
        df = self.preprocessing(test_dataset)
        dl = self.model.dls.test_dl(df)
        preds = self.model.get_preds(dl=dl)[0].numpy().flatten()
        df["rating"] = preds
        scores = create_overscores(df.values, test_dataset)
        return scores