import os
import numpy as np
import tensorflow as tf
from data import get_data, show_mnist_preds

tf.set_random_seed(1)

# load data
train_data, train_labels, eval_data, eval_labels = get_data(p=0.2)
print("train_data: {}".format(train_data.shape))
print("train_labels: {}".format(train_labels.shape))
print("eval_data: {}".format(eval_data.shape))
print("eval_labels: {}".format(eval_labels.shape))

input_size, _ = train_data.shape
n_classes, _  = train_labels.shape
n1            = 20 # number of neurons in hidden layer
n_epoch       = 1000

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
cross_entropy = -tf.reduce_sum(y * tf.log(logits), axis=0)
cross_entropy = tf.reduce_mean(cross_entropy)

# accuracy
y_hat = tf.argmax(logits, 0)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 0), y_hat), dtype=tf.float32))

# optimizer & training step
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_step = optimizer.minimize(cross_entropy)

with tf.Session() as sess:

    # init variables (weights, etc)
    sess.run(tf.global_variables_initializer())

    for i in range(n_epoch):

        # run one gradient descent and update step
        [_, loss] = sess.run([train_step, cross_entropy], feed_dict={X: train_data, y: train_labels})

        # evaluate accuracy on test values
        acc = accuracy.eval(feed_dict={X: eval_data, y: eval_labels})

        if i % 100 == 0 or i in [0, n_epoch-1]:
            print("""
                epoch: {}
                loss: {}
                test accuracy: {}
            """.format(i, loss, acc))

    print("Final accuracy on test set: {}".format(acc))

    # get the weights' value (numpy arrays)
    W1_value, b1_value, W2_value, b2_value = sess.run([W1, b1, W2, b2])

    # get predictions on new values (numpy array) and plot 10 of them
    y_hat_eval = y_hat.eval(feed_dict={X: eval_data})
    fig = show_mnist_preds(eval_data, np.argmax(eval_labels, axis=0), y_hat_eval, 10, random_seed=1)
    fig.show()
