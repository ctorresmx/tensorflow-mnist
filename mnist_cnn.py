#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#  Load the MNIST data set
mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Parameters
EPOCHS = 10
BATCH_SIZE = 100


def conv_layer(input, weight_shape, bias_shape):
    weight_init = tf.random_normal_initializer(stddev=0.1)
    conv_weights = tf.get_variable('W', weight_shape, initializer=weight_init)

    bias_init = tf.constant_initializer(value=0)
    conv_biases = tf.get_variable('b', bias_shape, initializer=bias_init)

    conv = tf.nn.conv2d(input, conv_weights, strides=[1, 1, 1, 1],
                        padding='SAME')

    return tf.nn.relu(tf.nn.bias_add(conv, conv_biases))


def pool_layer(input):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


def fc_layer(input, weight_shape, bias_shape):
    weight_init = tf.random_normal_initializer(stddev=0.1)
    fc_weights = tf.get_variable('W', weight_shape, initializer=weight_init)

    bias_init = tf.constant_initializer(value=0.1)
    fc_biases = tf.get_variable('b', bias_shape, initializer=bias_init)

    return tf.nn.relu(tf.matmul(input, fc_weights) + fc_biases)


def model(input, dropout_rate=1.0):

    with tf.variable_scope('conv_1'):
        conv_layer_1 = conv_layer(input, [5, 5, 1, 32], [32])
        max_pool_layer_1 = pool_layer(conv_layer_1)

    with tf.variable_scope('conv_2'):
        conv_layer_2 = conv_layer(max_pool_layer_1, [5, 5, 32, 64], [64])
        max_pool_layer_2 = pool_layer(conv_layer_2)

    with tf.variable_scope('fc_1'):
        reshape = tf.reshape(max_pool_layer_2, [-1, 49 * 64])
        hidden = fc_layer(reshape, [49 * 64, 512], [512])

    with tf.variable_scope('fc_2'):
        dropout = tf.nn.dropout(hidden, dropout_rate)
        output = fc_layer(dropout, [512, 10], [10])

    return output


def eval_in_batches(x, real_y):
    test_logits = model(x)
    correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(test_logits), 1),
                                  tf.argmax(real_y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    total_batch = int(mnist_data.test.num_examples / BATCH_SIZE)

    network_accuracy_total = 0
    for i in range(total_batch):
        batch_x, batch_y = mnist_data.test.next_batch(BATCH_SIZE)

        feed_dict = {
            x: np.reshape(batch_x, (-1, 28, 28, 1)),
            real_y: batch_y
        }

        network_accuracy_total += session.run(accuracy, feed_dict=feed_dict)

    return network_accuracy_total / total_batch


with tf.variable_scope("shared_variables") as scope:
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    real_y = tf.placeholder(tf.float32, shape=[None, 10])
    logits = model(x, 0.5)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=real_y,
                                                              logits=logits))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    with tf.Session() as session:
        tf.global_variables_initializer().run()

        for epoch in range(EPOCHS):
            total_batch = int(mnist_data.train.num_examples / BATCH_SIZE)

            for i in range(total_batch):
                batch_x, batch_y = mnist_data.train.next_batch(BATCH_SIZE)

                feed_dict = {
                    x: np.reshape(batch_x, (-1, 28, 28, 1)),
                    real_y: batch_y
                }

                session.run(optimizer, feed_dict=feed_dict)

            print('Epoch {}'.format(epoch))

        scope.reuse_variables()
        network_accuracy = eval_in_batches(x, real_y)

        print('The final accuracy over the MNIST data is {:2f}%'.format(network_accuracy * 100))

        print('Saving model...')
        saver = tf.train.Saver()
        saver.save(session, 'mnist_trained')

