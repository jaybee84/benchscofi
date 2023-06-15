#coding: utf-8

from stanscofi.models import BasicModel
import stanscofi.preprocessing

import numpy as np

import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input

class SimpleBinaryClassifier(BasicModel):
    def __init__(self, params=None):
        params = params if (params is not None) else self.default_parameters()
        super(SimpleBinaryClassifier, self).__init__(params)
        self.name = "SimpleBinaryClassifier"
        self.scalerP, self.scalerS = None, None
        for p in params:
            setattr(self, p, params[p])
        assert self.preprocessing_str in ["Perlman_procedure", "meanimputation_standardize", "same_feature_preprocessing"]
        self.filter = None
        assert len(self.layers_dims)>0

    def default_parameters(self):
        params = {
            "layers_dims": [16,32], "decision_threshold": 0.5, "preprocessing_str": "meanimputation_standardize", 
            "subset": None, "steps_per_epoch":1, "epochs":50,
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

    def _binary_crossentropy(self, log_ratio_p, log_ratio_q):
        loss_q, loss_p = [
            tf.nn.sigmoid_cross_entropy_with_logits(logits=x,labels=tf.ones_like(x) if (ix>0) else tf.zeros_like(x))
            for ix, x in enumerate([log_ratio_q, log_ratio_p])
        ]
        return tf.reduce_mean(loss_p + loss_q)
        
    def fit(self, train_dataset):
        X, y = self.preprocessing(train_dataset)
        log_ratio = Sequential(
            [Dense(self.layers_dims[0], input_dim=X.shape[1], activation='relu')]
            +[Dense(x, activation="relu") for x in self.layers_dims[1:]]
            +[Dense(1)]
        )
        x_p = Input(shape=(X.shape[1],))
        x_q = Input(shape=(X.shape[1],))
        log_ratio_p = log_ratio(x_p)
        log_ratio_q = log_ratio(x_p)
        self.model = Model(inputs=[x_p, x_q], outputs=[log_ratio_p, log_ratio_q])
        self.model.add_loss(self._binary_crossentropy(log_ratio_p, log_ratio_q))
        self.model.compile(optimizer='rmsprop', loss=None, metrics=['accuracy'])
        XX = [X[y==v,:] for v in [1,-1]]
        YY = [tf.ones(np.sum(y==1)), tf.zeros(np.sum(y==-1))]
        if (XX[0].shape[0]>XX[1].shape[0]):
            n = XX[0].shape[0]-XX[1].shape[0]
            XX[1] = np.concatenate(( XX[1], np.tile(XX[1][0,:],(n, 1)) ), axis=0)
            YY[1] = tf.zeros(np.sum(y==-1)+n)
        elif (XX[0].shape[0]<XX[1].shape[0]):
            n = XX[1].shape[0]-XX[0].shape[0]
            XX[0] = np.concatenate(( XX[0], np.tile(XX[0][0,:],(n, 1)) ), axis=0)
            YY[0] = tf.ones(np.sum(y==1)+n)
        hist = self.model.fit(x=XX, y=YY, steps_per_epoch=self.steps_per_epoch, epochs=self.epochs, verbose=0)

    def model_predict(self, test_dataset):
        X, y = self.preprocessing(test_dataset)
        p_pred, q_pred = self.model.predict(x=[X,X])
        preds = np.concatenate((tf.nn.sigmoid(p_pred).numpy(), tf.nn.sigmoid(q_pred).numpy()), axis=1)
        ids = np.argwhere(np.ones(test_dataset.ratings_mat.shape))
        predicted_ratings = np.zeros((X.shape[0], 3))
        predicted_ratings[:,0] = ids[:X.shape[0],1] 
        predicted_ratings[:,1] = ids[:X.shape[0],0] 
        predicted_ratings[:,2] = preds.max(axis=1)
        return predicted_ratings
