# -*- coding:utf-8 -*-

# Author: yaoC
# Created time: 2017年5月26日 下午2:39
# Email: chengyao09@hotmail.com
# Description: Restricted Boltzmann Machines (RBMs)

import tensorflow as tf
import numpy as np


class RBM(object):

    def __init__(self, input_size, output_size, epochs=5, learning_rate=0.1, batch_size=100):

        # 定义超参数
        self._input_size = input_size
        self._output_size = output_size
        # 训练迭代次数
        self._epochs = epochs
        # 学习速率
        self._learning_rate = learning_rate
        # 每次迭代使用的数据集大小
        self._batch_size = batch_size

        self.w = np.zeros([input_size, output_size],  np.float16)
        self.hb = np.zeros([output_size], np.float16)
        self.vb = np.zeros([input_size], np.float16)

    @staticmethod
    def prob_h_given_v(visible, w, hb):
        return tf.nn.sigmoid(tf.matmul(visible, w) + hb)

    @staticmethod
    def prob_v_given_h(hidden, w, vb):
        return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(w)) + vb)

    @staticmethod
    def sample_prob(prob):
        return tf.nn.relu(tf.sign(prob - tf.random_uniform(tf.shape(prob))))

    def train(self, x):

        _w = tf.placeholder("float", [self._input_size, self._output_size])
        _hb = tf.placeholder("float", [self._output_size])
        _vb = tf.placeholder("float", [self._input_size])

        prv_w = np.zeros([self._input_size, self._output_size], np.float16)
        prv_hb = np.zeros([self._output_size], np.float16)
        prv_vb = np.zeros([self._input_size], np.float16)

        cur_w = np.zeros([self._input_size, self._output_size], np.float16)
        cur_hb = np.zeros([self._output_size], np.float16)
        cur_vb = np.zeros([self._input_size], np.float16)
        v0 = tf.placeholder("float", [None, self._input_size])

        # Initialize with sample probabilities
        h0 = self.sample_prob(self.prob_h_given_v(v0, _w, _hb))
        v1 = self.sample_prob(self.prob_v_given_h(h0, _w, _vb))
        h1 = self.prob_h_given_v(v1, _w, _hb)

        # Create the Gradients
        positive_grad = tf.matmul(tf.transpose(v0), h0)
        negative_grad = tf.matmul(tf.transpose(v1), h1)

        # Update learning rates for the layers
        update_w = _w + self._learning_rate * (positive_grad - negative_grad) / tf.to_float(tf.shape(v0)[0])
        update_vb = _vb + self._learning_rate * tf.reduce_mean(v0 - v1, 0)
        update_hb = _hb + self._learning_rate * tf.reduce_mean(h0 - h1, 0)

        # Find the error rate
        err = tf.reduce_mean(tf.square(v0 - v1))

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables)
            # 迭代
            for epoch in range(self._epochs):
                for start, end in zip(range(0, len(x), self._batch_size),
                                      range(self._batch_size, len(x), self._batch_size)):
                    batch = x[start:end]
                    cur_w = sess.run(update_w, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    cur_hb = sess.run(update_hb, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    cur_vb = sess.run(update_vb, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    prv_w = cur_w
                    prv_hb = cur_hb
                    prv_vb = cur_vb
                error = sess.run(err, feed_dict={v0: x, _w: cur_w, _vb: cur_vb, _hb: cur_hb})
                print('Epoch: %d' % epoch, 'reconstruction error: %f' % error)
            self.w = prv_w
            self.hb = prv_hb
            self.vb = prv_vb

    def rbm_output(self, x):
        input_x = tf.constant(x)
        _w = tf.constant(self.w)
        _hb = tf.constant(self.hb)
        out = tf.nn.sigmoid(tf.matmul(input_x, _w) + _hb)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(out)
