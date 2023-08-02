#coding: utf-8

from benchscofi.SimpleBinaryClassifier import SimpleNeuralNetwork

import numpy as np

import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input

class SimplePULearning(SimpleNeuralNetwork):
    def __init__(self, params=None):
        params = params if (params is not None) else self.default_parameters()
        super(SimplePULearning, self).__init__(params)
        self.name = "SimplePULearning"

    def default_parameters(self):
        params = super(SimplePULearning, self).default_parameters()
        params.setdefault("PI", 0.33)
        return params

    def _custom_cross_entropy(self, log_ratio_p, labels):
        temp = tf.cast(tf.math.softplus(log_ratio_p), np.float32)
        weights = tf.cast(tf.constant([-1 / (self.PI - 1), (2 * self.PI - 1) / (self.PI - 1)]), np.float32)
        coef = tf.constant([0., 1.])
        bundle = temp[..., None] * weights[None, ...] - log_ratio_p[..., None] * coef[None, ...]
        oh = tf.one_hot(labels, depth=2)
        return tf.reduce_sum(tf.reduce_sum(bundle * oh, axis=0) / tf.reduce_sum(oh, axis=0))

    def nn_creation(self, x_p, x_q, log_ratio_p, log_ratio_q, X, y):
        labels = Input(shape=(1,), dtype=tf.int32)
        model = Model(inputs=[x_p, labels], outputs=log_ratio_p)
        model.add_loss(self._custom_cross_entropy(log_ratio_p, labels))
        model.compile(optimizer='rmsprop', loss=None, metrics=['accuracy'])
        XX = np.concatenate(tuple([X[y==v,:] for v in [1,-1]]), axis=0)
        YY = np.concatenate(
            (
                tf.ones(np.sum(y==1)), tf.zeros(np.sum(y==-1))
            )        
        ).astype(np.int32)
        return model, [XX, YY], YY

    def model_predict_proba(self, X):
        preds = self.model.predict(x=[X,tf.zeros(X.shape[0])])
        return np.ravel(tf.nn.sigmoid(preds).numpy())
