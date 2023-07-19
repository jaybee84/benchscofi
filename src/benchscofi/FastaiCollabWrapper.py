#coding: utf-8

from stanscofi.models import BasicModel
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
        self.name = "FastaiCollabWrapper"

    def default_parameters(self):
        params = {
            "n_iterations": 3, 
            "n_factors" :20,
            "weight_decay" : 0.1,
            "learning_rate" : 5e-3,
            "random_state": 1354,
        }
        return params

    def preprocessing(self, dataset, is_training=True):
        df = pd.DataFrame(dataset.ratings, index=range(dataset.ratings.shape[0]), columns=["disease","drug","rating"]).astype(int)
        return [df]
    
    def model_fit(self, df):
        np.random.seed(self.random_state)
        data = CollabDataLoaders.from_df(df, user_name='disease', item_name="drug", rating_name="rating", bs=64, seed=self.random_state, valid_pct=0.05)
        self.model = collab_learner(data, n_factors=self.n_factors, use_nn=True, y_range=self.y_range, emb_szs=None)
        self.model.fit_one_cycle(self.n_iterations, self.learning_rate, wd=self.weight_decay)
    
    def model_predict_proba(self, df):
        ## https://docs.fast.ai/tutorial.tabular
        dl = self.model.dls.test_dl(df)
        preds = self.model.get_preds(dl=dl)[0].numpy().flatten()
        return preds