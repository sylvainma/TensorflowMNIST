import numpy as np
import tensorflow as tf
from data import get_data

tf.set_random_seed(1)

# load data
train_data, train_labels, eval_data, eval_labels = get_data(p=0.5)

input_size, _ = train_data.shape
n_classes, _  = train_labels.shape
n1            = 20 # number of neurons in hidden layer
n_epoch       = 200

# input data in shape (#features, #examples)
X = tf.placeholder(tf.float32, shape=[input_size, None], name="X")
y = tf.placeholder(tf.float32, shape=[n_classes, None], name="y")

# hidden layer
W1 = tf.Variable(tf.random_normal([n1, input_size]), dtype=tf.float32, name="W1")
b1 = tf.Variable(tf.random_normal([n1, 1]), dtype=tf.float32, name="b1")
z1 = tf.matmul(W1, X) + b1
a1 = tf.tanh(z1)

# output layer
W2 = tf.Variable(tf.random_normal([n_classes, n1]), dtype=tf.float32, name="W2")
b2 = tf.Variable(tf.random_normal([n_classes, 1]), dtype=tf.float32, name="b2")
z2 = tf.matmul(W2, a1) + b2
logits = tf.nn.softmax(z2)

# cross-entropy loss
cross_entropy = -tf.reduce_sum(tf.multiply(y, tf.log(logits)))
cross_entropy = tf.reduce_mean(cross_entropy)

# optimizer & training step
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_step = optimizer.minimize(cross_entropy)

with tf.Session() as sess:

    # init variables (weights, etc)
    sess.run(tf.global_variables_initializer())

    for i in range(n_epoch):

        [_, loss] = sess.run([train_step, cross_entropy], feed_dict={X: train_data, y: train_labels})

        if i % 100 == 0 or i in [0, n_epoch-1]:
            print("""
                epoch: {}
                loss: {}
            """.format(i, loss))







