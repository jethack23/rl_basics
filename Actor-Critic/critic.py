import tensorflow as tf
import numpy as np

import tensorflow as tf
import numpy as np


def clipped_error(x):
    return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)


class Critic:

    def __init__(self, sess, config, env):
        self.sess = sess
        self.config = config
        self.env = env

    def build_net(self):
        with tf.variable_scope("qnet"):
            self.S_in = tf.placeholder(
                tf.float32, (None,) + self.config.input_shape, name="state")
            self.Q = self._fc_net(self.S_in)

        self.loss, self.train_op = self._loss_and_train_op()

        print("Critic Network Built")

    def _fc_net(self, s):
        initializer = tf.contrib.layers.xavier_initializer(uniform=False)
        activation_fn = tf.nn.relu

        h = s
        for i, dim in enumerate(self.config.q_fc_archi):
            h = tf.layers.dense(
                inputs=h,
                units=dim,
                activation=activation_fn,
                kernel_initializer=initializer,
                name="fc{}".format(i + 1))
        q = tf.layers.dense(
            inputs=h,
            units=self.env.action_space_size,
            kernel_initializer=initializer)
        return q

    def _loss_and_train_op(self):
        with tf.variable_scope("critic_optimizer"):
            self.A_in = tf.placeholder(tf.int32, (None,), name="action")
            self.target = tf.placeholder(tf.float32, (None,), name="target_q")
            a_one_hot = tf.one_hot(
                self.A_in, self.env.action_space_size, 1., 0., name="one_hot")
            self.Q_batch = tf.reduce_sum(
                tf.multiply(a_one_hot, self.Q), axis=1, name="q_batch")
            error = self.target - self.Q_batch
            loss = tf.reduce_mean(clipped_error(error), name="loss")
            train_op = tf.train.AdamOptimizer(self.config.lr_q).minimize(loss)
        return loss, train_op