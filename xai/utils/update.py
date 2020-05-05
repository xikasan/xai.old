# coding: utf-8

import tensorflow as tf
import tensorflow.keras as tk


def get_variables(net):
    if isinstance(net, tf.Variable):
        return net
    if isinstance(net, (tk.layers.Dense, tk.layers.Conv2D)):
        return net.trainable_variables
    if isinstance(net, tk.Model):
        return net.trainable_variables
    if hasattr(net, "trainable_variables"):
        return get_variables(net.trainable_variables)
    raise ValueError(
        "type of net must be tf.Variable, tf.keras.layers.Dense, tf.keras.layers.Conv2D, tf.keras.Model, "
        "or class which has trainable variables attribute, "
        "but {} is given".format(type(net))
    )


def copy(source, target):
    source = get_variables(source)
    target = get_variables(target)
    if not isinstance(source, list) and not isinstance(target, list):
        target.assign(source)
        return
    for s, t in zip(source, target):
        t.assign(s)


def soft_update(source, target, tau=0.01):
    source = get_variables(source)
    target = get_variables(target)
    if not isinstance(source, list) and not isinstance(target, list):
        target.assign((1 - tau) * target + tau * source)
        return
    for s, t in zip(source, target):
        t.assign((1 - tau) * t + tau * s)

