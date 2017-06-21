#!/usr/bin/env python

import tensorflow
from tensorflow.examples.tutorials.mnist import input_data

# Load the MNIST data set
mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_epochs = 20000
batch_size = 100


def weight_variable(shape):
    initial = tensorflow.truncated_normal(shape, stddev=0.1)
    return tensorflow.Variable(initial)


def bias_variable(shape):
    initial = tensorflow.constant(0.1, shape=shape)
    return tensorflow.Variable(initial)


def conv2d(x, W):
    return tensorflow.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tensorflow.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


real_y = tensorflow.placeholder(tensorflow.float32, [None, 10])

input = tensorflow.placeholder(tensorflow.float32, shape=[None, 784], name='input')
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

input_image = tensorflow.reshape(input, [-1, 28, 28, 1])
h_conv1 = tensorflow.nn.relu(conv2d(input_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tensorflow.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tensorflow.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tensorflow.nn.relu(tensorflow.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tensorflow.placeholder(tensorflow.float32)
h_fc1_drop = tensorflow.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tensorflow.nn.softmax(tensorflow.matmul(h_fc1_drop, W_fc2) + b_fc2, name='predict')

cross_entropy = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits(labels=real_y, logits=y_conv))

train_step = tensorflow.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tensorflow.equal(tensorflow.argmax(y_conv, 1), tensorflow.argmax(real_y, 1))

accuracy = tensorflow.reduce_mean(tensorflow.cast(correct_prediction, tensorflow.float32))

init = tensorflow.global_variables_initializer()

with tensorflow.Session() as session:
    session.run(init)

    for epoch in range(training_epochs):
        batch_x, batch_y = mnist_data.train.next_batch(batch_size)

        session.run(train_step, feed_dict={input: batch_x, real_y: batch_y, keep_prob: 0.5})
        print('Epoch {}'.format(epoch))

    correct_prediction = tensorflow.equal(tensorflow.argmax(y_conv, 1), tensorflow.argmax(real_y, 1))
    accuracy = tensorflow.reduce_mean(tensorflow.cast(correct_prediction, tensorflow.float32))

    network_accuracy = session.run(accuracy, feed_dict={input: mnist_data.test.images, real_y: mnist_data.test.labels, keep_prob: 1.0})

    print('The final accuracy over the MNIST data is {:.2f}%'.format(network_accuracy * 100))
