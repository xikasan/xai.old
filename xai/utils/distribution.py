# coding: utf-8

import numpy as np
import tensorflow as tf


class GaussianDistribution:

    def __init__(self):
        pass

    @staticmethod
    def sample(params):
        means = params["mean"]
        log_stds = params["log_std"]
        return means + tf.random.normal(shape=means.shape) * tf.math.exp(log_stds)

    @staticmethod
    def log_likelihood(x, params):
        means = params["mean"]
        log_stds = params["log_std"]
        assert means.shape == log_stds.shape
        zs = (x - means) / tf.exp(log_stds)
        return - tf.reduce_sum(log_stds, axis=-1) \
               - 0.5 * tf.reduce_sum(tf.square(zs), axis=-1) \
               - 0.5 * means.shape[1] * tf.math.log(2 * np.pi)

    @staticmethod
    def entropy(param):
        log_stds = param["log_std"]
        return tf.reduce_sum(log_stds + tf.math.log(tf.math.sqrt(2 * np.pi * np.e)), axis=-1)
