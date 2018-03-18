import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

def show_mnist_preds(X, y, y_hat, n, ncols=4, random_seed=False):
    """Plot mnist random images along with predicted label y_hat"""

    if random_seed and isinstance(random_seed, int):
        np.random.seed(random_seed)

    indices = np.random.choice(range(X.shape[0]), size=n, replace=False)
    nrows = int(np.ceil(n / ncols))

    fig, ax = plt.subplots(ncols=ncols, nrows=nrows)
    k = 0
    for i in range(nrows):
        for j in range(ncols):
            if k >= n:
                ax[i, j].imshow(np.zeros((28, 28)), cmap="gray")
            else:
                ax[i, j].imshow(np.reshape(X[:, indices[k]], (28, 28)), cmap="gray")
                ax[i, j].set(title="y: {} | y_hat: {}".format(
                                    y[indices[k]],
                                    y_hat[indices[k]]
                                )
                            )
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
            k = k + 1

    fig.tight_layout()
    return fig

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
        train_data = train_data[:, 0:n_train]
        train_labels = train_labels[:, 0:n_train]
        n_eval = int(p * eval_data.shape[1])
        eval_data = eval_data[:, 0:n_eval]
        eval_labels = eval_labels[:, 0:n_eval]

    return train_data, train_labels, eval_data, eval_labels

if __name__ == "__main__":

    train_data, train_labels, eval_data, eval_labels = get_data(p=0.5)

    assert (train_data.shape[0] == 784)
    assert (eval_data.shape[0] == 784)

    assert (train_labels.shape[0] == 10)
    assert (eval_labels.shape[0] == 10)

    y = np.argmax(train_labels, axis=0)
    fig = show_mnist_preds(train_data, y, y, 10)
    fig.show()
