import numpy as np
import tensorflow as tf
from data import get_data, show_mnist_preds

seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)

def fc_layer(input, input_size, units, activation=tf.nn.relu, name="fc_layer"):
    with tf.name_scope(name):
        W = tf.Variable(tf.random_normal([units, input_size]), dtype=tf.float32, name="W")
        b = tf.Variable(tf.random_normal([units, 1]), dtype=tf.float32, name="b")
        z = tf.matmul(W, input) + b
        return activation(z)

# load data
train_data, train_labels, eval_data, eval_labels = get_data(p=0.4)
print("train_data: {}".format(train_data.shape))
print("train_labels: {}".format(train_labels.shape))
print("eval_data: {}".format(eval_data.shape))
print("eval_labels: {}".format(eval_labels.shape))

input_size, _ = train_data.shape
n_classes, _  = train_labels.shape
n1            = 20 # number of neurons in hidden layer
n_epoch       = 5000

# input data in shape (#features, #examples)
X = tf.placeholder(tf.float32, shape=[input_size, None], name="X")
y = tf.placeholder(tf.float32, shape=[n_classes, None], name="y")

# hidden and output layers
hidden = fc_layer(X, input_size, n1)
logits = fc_layer(hidden, n1, n_classes, activation=tf.nn.softmax)

# cross-entropy loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits, dim=0))
tf.summary.scalar('cross entropy loss', cross_entropy)

# accuracy
y_hat = tf.argmax(logits, 0)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 0), y_hat), dtype=tf.float32))
tf.summary.scalar('train accuracy', accuracy)

# optimizer & training step
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_step = optimizer.minimize(cross_entropy)

with tf.Session() as sess:

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("./tmp/train", sess.graph)

    # init variables (weights, etc)
    sess.run(tf.global_variables_initializer())

    try:

        for i in range(n_epoch):

            # run one gradient descent and update step
            [summary, loss, _] = sess.run([merged, cross_entropy, train_step], feed_dict={X: train_data, y: train_labels})

            # evaluate accuracy on test values
            acc = accuracy.eval(feed_dict={X: eval_data, y: eval_labels})

            if i % 100 == 0 or i in [0, n_epoch-1]:
                train_writer.add_summary(summary, global_step=i) # bug avec acc
                print("""
                    epoch: {}
                    loss: {}
                    test accuracy: {}
                """.format(i, loss, acc))

        print("Final accuracy on test set: {}".format(acc))

    except KeyboardInterrupt:
        pass

    # get predictions on new values (numpy array) and plot 10 of them
    y_hat_eval = y_hat.eval(feed_dict={X: eval_data})
    fig = show_mnist_preds(eval_data, np.argmax(eval_labels, axis=0), y_hat_eval, 10, random_seed=1)
    fig.show()
