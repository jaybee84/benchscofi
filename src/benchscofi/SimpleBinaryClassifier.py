#coding: utf-8

from stanscofi.models import BasicModel
from stanscofi.preprocessing import preprocessing_XY

import numpy as np
import os
import random

import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input

class SimpleNeuralNetwork(BasicModel):
    def __init__(self, params=None):
        params = params if (params is not None) else self.default_parameters()
        super(SimpleNeuralNetwork, self).__init__(params)
        self.name = "SimpleNeuralNetwork"
        self.scalerP, self.scalerS, self.filter = None, None, None
        assert self.preprocessing_str in ["Perlman_procedure", "meanimputation_standardize", "same_feature_preprocessing"]
        assert len(self.layers_dims)>0

    def default_parameters(self):
        params = {
            "layers_dims": [16,32], "preprocessing_str": "meanimputation_standardize", 
            "subset": None, "steps_per_epoch":1, "epochs":50, "random_state": 1234,
        }
        return params

    def preprocessing(self, dataset, is_training=True):
        X, y, scalerS, scalerP, filter_ = preprocessing_XY(dataset, self.preprocessing_str, subset_=self.subset, filter_=self.filter, scalerS=self.scalerS, scalerP=self.scalerP, inf=2, njobs=1)
        X = np.nan_to_num(X, nan=0.)
        self.filter = filter_
        self.scalerS = scalerS
        self.scalerP = scalerP
        return [X, y] if (is_training) else [X]
        
    def model_fit(self, X, y):
        os.environ['PYTHONHASHSEED']=str(self.random_state)
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # new flag present in tf 2.0+
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        tf.random.set_seed(self.random_state)
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        log_ratio = Sequential(
            [Dense(self.layers_dims[0], input_dim=X.shape[1], activation='relu')]
            +[Dense(x, activation="relu") for x in self.layers_dims[1:]]
            +[Dense(1)]
        )
        x_p = Input(shape=(X.shape[1],))
        x_q = Input(shape=(X.shape[1],))
        log_ratio_p = log_ratio(x_p)
        log_ratio_q = log_ratio(x_p)
        self.model, XX, YY = self.nn_creation(x_p, x_q, log_ratio_p, log_ratio_q, X, y)
        hist = self.model.fit(x=XX, y=YY, steps_per_epoch=self.steps_per_epoch, epochs=self.epochs, verbose=0)

    def model_predict_proba(self, X):
        raise NotImplemented

    def nn_creation(self, x_p, x_q, log_ratio_p, log_ratio_q, X, y):
        raise NotImplemented

class SimpleBinaryClassifier(SimpleNeuralNetwork):
    def __init__(self, params=None):
        params = params if (params is not None) else self.default_parameters()
        super(SimpleBinaryClassifier, self).__init__(params)
        self.name = "SimpleBinaryClassifier"

    def _binary_crossentropy(self, log_ratio_p, log_ratio_q):
        loss_q, loss_p = [
            tf.nn.sigmoid_cross_entropy_with_logits(logits=x,labels=tf.ones_like(x) if (ix>0) else tf.zeros_like(x))
            for ix, x in enumerate([log_ratio_q, log_ratio_p])
        ]
        return tf.reduce_mean(loss_p + loss_q)

    def nn_creation(self, x_p, x_q, log_ratio_p, log_ratio_q, X, y):
        model = Model(inputs=[x_p, x_q], outputs=[log_ratio_p, log_ratio_q])
        model.add_loss(self._binary_crossentropy(log_ratio_p, log_ratio_q))
        model.compile(optimizer='rmsprop', loss=None, metrics=['accuracy'])
        XX = [X[y==1,:], X[y<1]]
        YY = [tf.ones(np.sum(y==1)), tf.zeros(np.sum(y<1))]
        if (XX[0].shape[0]>XX[1].shape[0]):
            n = XX[0].shape[0]-XX[1].shape[0]
            XX[1] = np.concatenate(( XX[1], np.tile(XX[1][0,:],(n, 1)) ), axis=0)
            YY[1] = tf.zeros(np.sum(y<1)+n)
        elif (XX[0].shape[0]<XX[1].shape[0]):
            n = XX[1].shape[0]-XX[0].shape[0]
            XX[0] = np.concatenate(( XX[0], np.tile(XX[0][0,:],(n, 1)) ), axis=0)
            YY[0] = tf.ones(np.sum(y==1)+n)
        return model, XX, YY

    def model_predict_proba(self, X):
        p_pred, q_pred = self.model.predict(x=[X, X])
        preds = np.concatenate((tf.nn.sigmoid(p_pred).numpy(), tf.nn.sigmoid(q_pred).numpy()), axis=1)
        return preds[:,1]
        
