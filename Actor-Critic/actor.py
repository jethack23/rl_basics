import tensorflow as tf
import numpy as np


class Actor:

    def __init__(self, sess, config, env):
        self.sess = sess
        self.config = config
        self.env = env

    def build_net(self):
        with tf.variable_scope("policy"):
            self.S_in = tf.placeholder(
                tf.float32, (None,) + self.config.input_shape, name="state")
            self.pi = self._fc_net(self.S_in)

        self.loss, self.train_op = self._loss_and_train_op()

        print("Actor Network Built")

    def _fc_net(self, s):
        initializer = tf.contrib.layers.xavier_initializer(uniform=False)
        activation_fn = tf.nn.relu

        h = s
        for i, dim in enumerate(self.config.pi_fc_archi):
            h = tf.layers.dense(
                inputs=h,
                units=dim,
                activation=activation_fn,
                kernel_initializer=initializer,
                name="fc{}".format(i + 1))
        h = tf.layers.dense(
            inputs=h,
            units=self.env.action_space_size,
            kernel_initializer=initializer)
        pi = tf.nn.softmax(h, name="pi")
        return pi

    def choose_action(self):
        pi = self.sess.run(self.pi, {self.S_in: [self.env.state]})[0]
        a = np.random.choice(self.env.action_space_size, p=pi)
        return a

    def _loss_and_train_op(self):
        with tf.variable_scope("actor_optimizer"):
            self.A_in = tf.placeholder(tf.int32, (None,), name="action")
            a_one_hot = tf.one_hot(
                self.A_in, self.env.action_space_size, 1., 0., name="a_one_hot")
            self.Q_in = tf.placeholder(tf.float32, (None,), name="q")
            self.pi_s_a = tf.reduce_sum(
                tf.multiply(a_one_hot, self.pi), axis=1, name="pi_s_a")

            loss = tf.reduce_sum(-self.Q_in * tf.log(self.pi_s_a), name="loss")

            train_op = tf.train.GradientDescentOptimizer(
                self.config.lr_pi).minimize(loss)

        return loss, train_op
