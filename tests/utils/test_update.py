# coding: utf-8

import numpy as np
import xtools as xt
import tensorflow as tf
from xai.utils.update import *
from xai.algorithms.dqn import QNet


def test_get_variables():
    print("[test] get_variables")
    dummy_input = np.zeros((2, 1)).astype(np.float32)

    test_var = tf.Variable([0, 1])
    result = get_variables(test_var)
    xt.info("test_var", result)

    test_dense = tf.keras.layers.Dense(2)
    test_dense(dummy_input)
    result = get_variables(test_dense)
    xt.info(result)

    test_model = QNet([2, 2], 2, 1)
    test_model(dummy_input)
    result = get_variables(test_model)
    xt.info(result)

    try:
        test_non_variable = np.zeros((10, 5))
        result = get_variables(test_non_variable)
    except ValueError as e:
        print(e)


def test_copy():
    print("[test] copy")
    dummy_input = np.zeros((2, 1)).astype(np.float32)

    source = tf.Variable(1)
    target = tf.Variable(0)
    copy(source, target)
    xt.info(source, target)

    print("- "*30)
    test_source = tf.keras.layers.Dense(2)
    test_target = tf.keras.layers.Dense(2)
    test_source(dummy_input)
    test_target(dummy_input)
    copy(test_source, test_target)
    xt.info(test_source.trainable_variables)
    xt.info(test_target.trainable_variables)


def test_soft_update():
    print("[test] soft_update")
    dummy_input = np.zeros((2, 1)).astype(np.float32)

    source = tf.Variable(1.0)
    target = tf.Variable(0.0)
    soft_update(source, target)
    xt.info(source, target)

    print("- "*30)
    test_source = tf.keras.layers.Dense(2)
    test_target = tf.keras.layers.Dense(2)
    test_source(dummy_input)
    test_target(dummy_input)
    xt.info(test_source.trainable_variables)
    xt.info(test_target.trainable_variables)
    for _ in range(10):
        soft_update(test_source, test_target, 0.1)
        xt.info(test_target.trainable_variables)


if __name__ == '__main__':
    test_get_variables()
    print("="*60)
    test_copy()
    print("="*60)
    test_soft_update()
