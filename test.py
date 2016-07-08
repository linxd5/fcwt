#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import numpy as np

num_points = 1000
vector_set = []
for i in xrange(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vector_set.append([x1, y1])

x_data = [v[0] for v in vector_set]
y_data = [v[1] for v in vector_set]


W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b
loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()

def animate(i):
    fig.clear()
    sess.run(train)
    plt.plot(x_data, y_data, 'ro', label='Original data')
    plt.plot(x_data, sess.run(W) * x_data + sess.run(b), 'b')

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
