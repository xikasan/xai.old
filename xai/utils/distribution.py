# coding: utf-8

import numpy as np
import tensorflow as tf


class Distribution:

    EPS = 1e-8

    @classmethod
    def entropy(cls, dist):
        raise NotImplementedError

    @classmethod
    def sample(cls, dist):
        raise NotImplementedError

    @classmethod
    def log_likehood(cls, x, dist):
        raise NotImplementedError

    @classmethod
    def kl(cls, dist1, dist2):
        raise NotImplementedError


class CategoricalDistribution(Distribution):

    @classmethod
    def sample(cls, prob):
        return tf.random.categorical(tf.math.log(prob), 1)

    @classmethod
    def log_likehood(cls, x, prob):
        assert x.shape == prob.shape,\
            "shape of x in one-hot {} and prob {} is not much".format(x.shape, prob.shape)
        return tf.math.log(tf.reduce_sum(x * prob, axis=1) + cls.EPS)

    @classmethod
    def entropy(cls, prob):
        return -tf.reduce_sum(prob * tf.math.log(prob + cls.EPS), axis=1)

    @classmethod
    def kl(cls, prob1, prob2):
        return tf.reduce_sum(
            prob1 * (tf.math.log(prob1 + cls.EPS) - tf.math.log(prob2 + cls.EPS))
        )
