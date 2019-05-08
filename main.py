#! /usr/bin/env python
"""Use neural networks for ILC."""
import tensorflow as tf
import tensorflow.contrib.layers as lays
import numpy as np
import matplotlib.pyplot as plt
import logging
tf.logging.set_verbosity(tf.logging.INFO)
n_classes = 1
res = 7
hm_epochs = 20
batch_size = 1000


def fcc(inputs):
    """Network Model."""
    net = lays.fully_connected(inputs, 3, activation_fn=tf.identity)
    net = lays.fully_connected(net, n_classes, activation_fn=tf.identity)
    return net


def train_neural_network(n):
    """
    Train the neural network model.

    Parameters
    ----------
    Input array,
    Cluster Number

    Returns
    -------
    None

    Raises
    ------
    None

    See Also
    --------
    fcc.neural_network_model()

    Notes
    -----
    None

    """
    train_x = np.loadtxt(f"./cluster_1/cluster_maps_{n}")
    logging.debug(train_x)
    logging.debug(train_x.shape)
    x = tf.placeholder('float', [None, res])
    y = tf.placeholder('float', [None, res])
    prediction = fcc(x)
    varss = tf.trainable_variables()
    lossl2 = tf.add_n([tf.nn.l2_loss(v) for v in varss])
    sqmean = tf.reduce_mean(tf.square(prediction))
    var = sqmean
    constraint = 100*tf.abs(fcc(y)[0][0] - 1)
    cost = constraint + lossl2 + var
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    epoch_data = []
    loss_data = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])

                _, c = sess.run([optimizer, cost],
                                feed_dict={x: batch_x,
                                           y: [np.ones(res)]})
                epoch_loss += c
                i += batch_size
            loss_data.append(epoch_loss)
            epoch_data.append(epoch)
            logging.info(f"Epoch {epoch+1}/{hm_epochs} "
                         f"loss: {epoch_loss}")
        maps = sess.run([prediction], feed_dict={x: train_x})[0]
        np.savetxt(f"final_map_{n}", maps)
    plt.plot(epoch_data, loss_data)
    plt.title(f"Epoch Loss for cluster {n}")
    plt.savefig(f"epoch_loss_{n}.png")
    plt.close()


def handle_exception(exc_type, exc_value, exc_traceback):
    """Handle Exceptions in log."""
    if issubclass(exc_type, KeyboardInterrupt):
        logging.error("Keyboard Interrupt",
                      exc_info=(exc_type, exc_value, exc_traceback))
        return

    logging.error("Uncaught exception",
                  exc_info=(exc_type, exc_value, exc_traceback))


if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", help="Set Log Level", type=str,
                        default='INFO')
    args = parser.parse_args()
    loglevel = args.log.upper()
    logging.getLogger('tensorflow')
    logging.basicConfig(format='(%(asctime)s %(filename)s %(levelname)s) '
                        + '%(funcName)s %(lineno)d >> %(message)s',
                        level=getattr(logging, loglevel, None))
    logging.captureWarnings(True)
    sys.excepthook = handle_exception
    for kk in range(13):
        train_neural_network(kk)
