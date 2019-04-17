import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from actor import Actor
from critic import Critic


class Agent:

    def __init__(self, sess, config, env):
        self.sess = sess
        self.config = config
        self.env = env

        self.actor = Actor(sess, config, env)
        self.critic = Critic(sess, config, env)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self._build_nets()

    def train(self):
        rewards = []
        avg_n = self.config.max_ep / 100
        total = 0
        for i in range(1, self.config.max_ep + 1):
            ep_reward = 0.
            done = False
            state = self.env.initialize_game()
            action = self.actor.choose_action()
            while not done:
                prev_state = state
                prev_action = action
                state, reward, done = self.env.act(action)
                action = self.actor.choose_action()
                ep_reward += reward
                self.after_act(prev_state, prev_action, reward, state, done,
                               action)
            total += ep_reward
            if i % avg_n == 0:
                rewards.append(total / avg_n)
                print("Episode: {}, Reward: {}".format(i, total / avg_n))
                total = 0
        plt.plot(rewards)
        plt.show()

    def after_act(self, prev, action, reward, state, done, next_action):
        self.update_params(prev, action, reward, state, done, next_action)

    def update_params(self, prev, action, reward, state, done, next_action):
        if not done:
            next_q = self.sess.run(self.critic.Q_batch, {
                self.critic.S_in: [state],
                self.critic.A_in: [next_action]
            })
            target = reward + self.config.df * next_q
        else:
            target = [reward]
        _, q_a = self.sess.run(
            (self.critic.train_op, self.critic.Q_batch), {
                self.critic.S_in: [prev],
                self.critic.A_in: [action],
                self.critic.target: target
            })
        self.sess.run(
            self.actor.train_op, {
                self.actor.S_in: [prev],
                self.actor.A_in: [action],
                self.actor.Q_in: q_a
            })

    def _build_nets(self):
        with tf.variable_scope("actor"):
            self.actor.build_net()
        with tf.variable_scope("critic"):
            self.critic.build_net()
        self.sess.run(tf.global_variables_initializer())
        print("Variables Initialized")
