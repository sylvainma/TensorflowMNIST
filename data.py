import numpy as np
import tensorflow as tf

def get_one_hot(targets, nb_classes):
    """Input a numpy array of values and output its one hot encoding"""
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]

def get_data(p=1):
    """Return MNIST dataset under shape convention (#features, #examples)"""

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    # train dataset
    train_data = mnist.train.images # Returns np.array
    train_data = train_data.T
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    train_labels = get_one_hot(train_labels, 10)
    train_labels = train_labels.T

    # test dataset
    eval_data = mnist.test.images # Returns np.array
    eval_data = eval_data.T
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    eval_labels = get_one_hot(eval_labels, 10)
    eval_labels = eval_labels.T

    if 0 < p < 1:
        n_train = int(p * train_data.shape[1])
        train_data = train_labels[:, 0:n_train]
        train_labels = train_labels[:, 0:n_train]
        n_eval = int(p * eval_data.shape[1])
        eval_data = eval_labels[:, 0:n_eval]
        eval_labels = eval_labels[:, 0:n_eval]

    return train_data, train_labels, eval_data, eval_labels

if __name__ == "__main__":

    train_data, train_labels, eval_data, eval_labels = get_data()

    assert (train_data.shape[0] == 784)
    assert (eval_data.shape[0] == 784)

    assert (train_labels.shape[0] == 10)
    assert (eval_labels.shape[0] == 10)