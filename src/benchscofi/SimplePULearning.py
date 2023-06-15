#coding: utf-8

from stanscofi.models import BasicModel
import stanscofi.preprocessing

import numpy as np

import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input

class SimplePULearning(BasicModel):
    def __init__(self, params=None):
        params = params if (params is not None) else self.default_parameters()
        super(SimplePULearning, self).__init__(params)
        self.name = "SimplePULearning"
        self.scalerP, self.scalerS = None, None
        for p in params:
            setattr(self, p, params[p])
        assert self.preprocessing_str in ["Perlman_procedure", "meanimputation_standardize", "same_feature_preprocessing"]
        self.filter = None
        assert len(self.layers_dims)>0

    def default_parameters(self):
        params = {
            "layers_dims": [16,32], "decision_threshold": 0.5, "preprocessing_str": "meanimputation_standardize", 
            "subset": None, "steps_per_epoch":1, "epochs":50, "PI": 0.33,
        }
        return params

    def preprocessing(self, dataset):
        if (self.preprocessing_str == "Perlman_procedure"):
            X, y = eval("stanscofi.preprocessing."+self.preprocessing_str)(dataset, njobs=1, sep_feature="-", missing=-666, verbose=False)
            scalerS, scalerP = None, None
        if (self.preprocessing_str == "meanimputation_standardize"):
            X, y, scalerS, scalerP = eval("stanscofi.preprocessing."+self.preprocessing_str)(dataset, subset=self.subset, scalerS=self.scalerS, scalerP=self.scalerP, inf=2, verbose=False)
        if (self.preprocessing_str == "same_feature_preprocessing"):
            X, y = eval("stanscofi.preprocessing."+self.preprocessing_str)(dataset)
            scalerS, scalerP = None, None
        if (self.preprocessing_str != "meanimputation_standardize"):
            if ((self.subset is not None) or (self.filter is not None)):
                if ((self.subset is not None) and (self.filter is None)):
                    with np.errstate(over="ignore"):
                        x_vars = [np.nanvar(X[:,i]) if (np.sum(~np.isnan(X[:,i]))>0) else 0 for i in range(X.shape[1])]
                        x_vars = [x if (not np.isnan(x) and not np.isinf(x)) else 0 for x in x_vars]
                        x_ids_vars = np.argsort(x_vars).tolist()
                        features = x_ids_vars[-self.subset:]
                        self.filter = features
                X = X[:,self.filter]
        self.scalerS = scalerS
        self.scalerP = scalerP
        return X, y

    def _custom_cross_entropy(self, log_ratio_p, labels):
        temp = tf.math.softplus(log_ratio_p)
        weights = tf.constant([-1 / (self.PI - 1), (2 * self.PI - 1) / (self.PI - 1)])
        coef = tf.constant([0., 1.])
        bundle = temp[..., None] * weights[None, ...] - log_ratio_p[..., None] * coef[None, ...]
        oh = tf.one_hot(labels, depth=2)
        return tf.reduce_sum(tf.reduce_sum(bundle * oh, axis=0) / tf.reduce_sum(oh, axis=0))
        
    def fit(self, train_dataset):
        X, y = self.preprocessing(train_dataset)
        labels = Input(shape=(1,), dtype=tf.int32)
        log_ratio = Sequential(
            [Dense(self.layers_dims[0], input_dim=X.shape[1], activation='relu')]
            +[Dense(x, activation="relu") for x in self.layers_dims[1:]]
            +[Dense(1)]
        )
        x_p = Input(shape=(X.shape[1],))
        log_ratio_p = log_ratio(x_p)
        self.model = Model(inputs=[x_p, labels], outputs=log_ratio_p)
        self.model.add_loss(self._custom_cross_entropy(log_ratio_p, labels))
        self.model.compile(optimizer='rmsprop', loss=None, metrics=['accuracy'])
        XX = np.concatenate(tuple([X[y==v,:] for v in [1,-1]]), axis=0)
        YY = np.concatenate(
            (
                tf.ones(np.sum(y==1)), tf.zeros(np.sum(y==-1))
            )        
        ).astype(np.int32)
        hist = self.model.fit(x=[XX,YY], y=YY, steps_per_epoch=self.steps_per_epoch, epochs=self.epochs, verbose=0)

    def model_predict(self, test_dataset):
        X, y = self.preprocessing(test_dataset)
        XX = X
        YY = tf.zeros(X.shape[0])
        pred = self.model.predict(x=[XX,YY])
        preds = np.ravel(tf.nn.sigmoid(pred).numpy())
        ids = np.argwhere(np.ones(test_dataset.ratings_mat.shape))
        predicted_ratings = np.zeros((X.shape[0], 3))
        predicted_ratings[:,0] = ids[:X.shape[0],1] 
        predicted_ratings[:,1] = ids[:X.shape[0],0] 
        predicted_ratings[:,2] = preds
        return predicted_ratings
