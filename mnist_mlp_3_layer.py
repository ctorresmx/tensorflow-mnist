#!/usr/bin/env python

#
#   Based on the example: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/multilayer_perceptron.py
#

import tensorflow
from tensorflow.examples.tutorials.mnist import input_data

# Load the MNIST data set
mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# Construct the graph
# First layer
input = tensorflow.placeholder(tensorflow.float32, shape=[None, 784], name='input')
w1 = tensorflow.Variable(tensorflow.random_normal([784, 256]))
b1 = tensorflow.Variable(tensorflow.random_normal([256]))
x2 = tensorflow.nn.relu(tensorflow.add(tensorflow.matmul(input, w1), b1))

# Second layer
w2 = tensorflow.Variable(tensorflow.random_normal([256, 256]))
b2 = tensorflow.Variable(tensorflow.random_normal([256]))
x3 = tensorflow.nn.relu(tensorflow.add(tensorflow.matmul(x2, w2), b2))

# Third layer
w3 = tensorflow.Variable(tensorflow.random_normal([256, 10]))
b3 = tensorflow.Variable(tensorflow.random_normal([10]))
y = tensorflow.matmul(x3, w3) + b3

# Output to be used when running with the UI
output = tensorflow.nn.softmax(y, name='predict')

# Placeholder for the correct answer
real_y = tensorflow.placeholder(tensorflow.float32, [None, 10])

# Loss function
cost = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits(logits=y, labels=real_y))

# Optimiation
optimizer = tensorflow.train.AdamOptimizer(learning_rate=learning_rate)
train_step = optimizer.minimize(cost)

# Initialization
init = tensorflow.global_variables_initializer()

with tensorflow.Session() as session:
    session.run(init)

    for epoch in range(training_epochs):
        total_batch = int(mnist_data.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_x, batch_y = mnist_data.train.next_batch(batch_size)

            session.run(train_step, feed_dict={input: batch_x, real_y: batch_y})

        print('Epoch {}'.format(epoch))

    correct_prediction = tensorflow.equal(tensorflow.argmax(y, 1), tensorflow.argmax(real_y, 1))
    accuracy = tensorflow.reduce_mean(tensorflow.cast(correct_prediction, tensorflow.float32))

    network_accuracy = session.run(accuracy, feed_dict={input: mnist_data.test.images, real_y: mnist_data.test.labels})

    print('The final accuracy over the MNIST data is {:.2f}%'.format(network_accuracy * 100))

    print('Saving model...')
    saver = tensorflow.train.Saver()
    saver.save(session, 'mnist_trained')
