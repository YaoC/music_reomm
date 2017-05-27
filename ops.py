# -*- coding:utf-8 -*-

# Author: ChengYao
# Created time: 2017年5月23日 下午2:11
# Email: chengyao09@hotmail.com
# Description:


import tensorflow as tf
import utils


def inferrence_model(user_batch, music_batch, user_num, music_num, dim=50):
    w_user = tf.get_variable('user_matrix', shape=[user_num, dim], dtype='float16',
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
    beta_user = tf.nn.embedding_lookup(w_user, user_batch, name="beta_user")

    dbn_w = []
    dbn_hb = []
    input_size, rbm_hidden_sizes, params = utils.load_params()
    for i in range(len(rbm_hidden_sizes)):
        dbn_w.append(tf.get_variable("dbn_w%d" % i, dtype='float16', initializer=params[i]['w']))
        dbn_hb.append(tf.get_variable("dbn_hb%d" % i, dtype='float16', initializer=params[i]['hb']))
    dbn = music_batch
    for w, hb in zip(dbn_w, dbn_hb):
        dbn = tf.add(tf.matmul(dbn, w), hb)

    infer = tf.reduce_sum(tf.multiply(beta_user, dbn), 1, name="result_inference")
    regularizer = tf.nn.l2_loss(beta_user, name="user_regularizer")

    return infer, regularizer


def optimization(infer, regularizer, rate_batch, learning_rate=0.001, reg=0.1, device="/cpu:0"):
    global_step = tf.train.get_global_step()
    assert global_step is not None
    with tf.device(device):
        cost_l2 = tf.nn.l2_loss(tf.subtract(infer, rate_batch))
        penalty = tf.constant(reg, dtype=tf.float16, shape=[], name="l2")
        cost = tf.add(cost_l2, tf.multiply(regularizer, penalty))
        train_op = tf.train.AdamOptimizer(
            learning_rate).minimize(cost, global_step=global_step)
    return cost, train_op
