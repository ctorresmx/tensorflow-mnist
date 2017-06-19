# TensorFlow 101 pt 2 - MNIST interface, Save models, 3-layer MLP

Last time I talked about TensorFlow basics and as an example I used a small neural network using a single layer with no real testing. This time I want to try something a little bit more graphical and bigger, a neural network with 3 layers and a simple interface to draw some numbers for it to classify.

First I am going to talk about how we can structure our code to fit this new testing UI, this will require us to talk about how to save and load models in TensorFlow. Finally, I will go through this time's MNIST model, it will use 3 layers (as opposed to just one) and some enhancements for trainning.

## MNIST UI

The reason behind this UI is to be able to test our MNIST trained model. Last time the script only show the accuracy prediction, but we couldn't actually draw numbers on the screen, so we had to believe that it was over 90% correct.

How this all fits in a TensorFlow model? Well, usually you train your model and when you achieve a good-enough accuracy rate you save your model, then in whatever application, web app or mobile app you need to use that model, you just load the saved file. This is how MNIST interface is going to be used, we are going to train our model, as we previously did, then save it. Later, MNIST UI will load that model and run predictions on it.

The UI comprises of 2 windows. One is a window with a 28x28 grid, which is the right size for our input samples. The second window is no other than a label showing which number the model predicts our drawed number is.

**Put screenshots here**

The code uses Python's TKinter, which is the easiest framework to use inside Python (some distributions have Tkinter already installed). It has only been tested on `macOS`, but since it's Python, it should be ready to be used on `Linux` and `Windows` as well. I am not going to give a detailed version of how all the code works, but this is a short resume of what is happening on the inside:

1. UI is launched and start listening for events on the 28x28 grid.
2. TensorFlow model is loaded from a file named `mnist_trained.meta`.
3. You draw a number in the grid while you keep the mouse button pressed.
4. When you release the button the grid will convert this 2d grid into a 1d array.
5. The 1d array is input into the model and returns a predicted class.
6. The predicted number is showned at the other window.

This code can be reused with any MNIST model, as long as the model is saved with the name `mnist_trained.meta`. Be aware that MNIST dataset is a collection of grayscale digitized numbers, while this code creates a 28x28 black and white grid, so it's not quite the same and results may not reflect the actual model accuracy, but it's enough for quick testing of your model.

All this code is inside the `main.py` file. You just run it and you will see the UI, but before we need to have a trained model ready to be loaded.

## Save TensorFlow models

TensorFlow has built-in functions to save and load models. Remember the graph TensorFlow builds to know what operations to run against which data? Well, that's exactly what our saved model will keep, as well as the weights of the neural network.

### Saving a model

A saved model will store the graph itself along side the weights and functions. This functions can be really any of the many TensorFlow operations we declared. In our case we just need a way to input data and call the prediction.

All TensorFlow operations and variables have a keyword argument `name` to identify them inside the graph. It is enough to mark our input placeholder and our prediction function with a `input` and `predict` labels.

```python
x = tensorflow.placeholder(tensorflow.float32, shape=[None, 784], name='input')
W = tensorflow.Variable(tensorflow.zeros([784, 10]))
b = tensorflow.Variable(tensorflow.zeros([10]))
y = tensorflow.nn.softmax(tensorflow.matmul(x, W) + b, name='predict')
```

As you can see, we only marked `x` and `y`, since the weights and biases are not really important for us to identify them later.

The easiest way to save a model is using a Saver:

```python
saver = tensorflow.train.Saver()
saver.save(session, 'mnist_trained')
```

`mnist_trained` is the name of the output files. These files are:

- **mnist_trained.meta:** The TensorFlow graph
- **mnist_trained.index:** The index for the checkpoints
- **mnist_trained.data-XXXXXX-of-XXXXXX:** A point in time for the checkpoints
- **checkpoint:** The latest checkpoint

The main file we are going to work with is the `mnist_trained.meta`, as this contains our model and the functions we are going to use (mainly for inputing data and predicting).

### Loading a model

On the `main.py` file, on function `load_model()` you can see that we are going to create a TensorFlow session. Then we are going to import the meta graph using the path to the model, in this case is the same working directory `.`, then we restore the checkpoint.

TensorFlow gives you the opportunity to save checkpoints, so that you can work with your model across time. You could potentially create a checkpoint every x epochs and see how it progresses over time, for the sake of this example we just have one checkpoint, so we load the latest checkpoint. Finally, we need to load our variables, this so we can work in the data from the UI, make predictions and of course, let TensorFlow know which graph to use.

```python
def load_model(self, model_path, latest_checkpoint='./'):
    """ Load the TensorFlow model.

    You can choose a given checkpoint, otherwise it will be the latests one.
    """

    # Creates a TensorFlow session
    self.tensorflow_session = tensorflow.Session()

    # Imports the Graph
    saver = tensorflow.train.import_meta_graph(model_path)

    # Loads the given checkpoint
    saver.restore(
        self.tensorflow_session,
        tensorflow.train.latest_checkpoint(latest_checkpoint)
    )

    # Loads the variables we are going to use, i.e., inputs and predictor
    # function.
    self.graph = tensorflow.get_default_graph()
    self.input = self.graph.get_tensor_by_name('input:0')
    self.predict = self.graph.get_tensor_by_name('predict:0')
```

The `:0` appended at the end of the name of the tensor/variable/function needs to be there, even though we only have one of those variables, this is per TensorFlow implementation.

Now, you can run the `mnist_mlp.py` file, it will train the same way the old code did, but this time you should see a bunch of `mnist_trained.*` files along side a `checkpoint` file. After that, you can launch `main.py`, you will see the grid, and you will be able to draw and predict a number.

**Insert screenshot of UI working**

## 3-layer MLP for MNIST

Last time the neural network used was a single-layered network. This time I want to try with a 3-layered one. This code is inside the `mnist_mlp_3_layer.py` file.

Traditionally, neural networks have 3 layers: input layer, hidden layer and output layer.

**Insert image of a 3 layer net**

The script starts the same, importing the TensorFlow library along side the MNIST data set.

```python
import tensorflow
from tensorflow.examples.tutorials.mnist import input_data

# Load the MNIST data set
mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

### Parameters

Before we start defining our graph, let's define some variables we are going to use:

```python
# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
```

The `learning rate` defines the size of the steps the optimizer is taking towards the minimum (to minimize the error). The `training epochs` are the amount of times our network will see the whole training set. The `batch size` is how many samples we are going to show to the network at once, i.e., in each epoch we will loop through all the samples and show `batch size` number of samples at once.

### Building the graph

In this case we want to define 3 layers:

- Input layer: Receives the input directly.
- Hidden layer: Receives input from the previous layer.
- Output layer: Recieves input from the hidden layer and then outputs the likehood for a given number between 0-9.

```python
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
```

The first layer is going to be a 784 input array, remember that the `None` in that placeholder means that we don't know yet how many 784 1d-array we are going to input at any given moment. Instead of having a weight matrix of 784x10, we are now going to use a 784x256 matrix, this will reduce the dimensionality of the previous layer, but will allow the network to keep extracting features in the next layer. We also set the name to `input`, which will come in handy when we test this using our UI.

The second layer is going to be a 256x256 matrix, there is not much else to it.

The third layer is a 256x10 weight matrix, which will reflect the probability of a given input into the range of digits between 0 and 9.

Finally, the output of the whole network is the softmax function applied to the output of the previous layer. We will call this `predict`, since this is the function that we will use in the UI to predict a given input.

Remember to set the `real_y` placeholder, this is going to be used at training time to show the network the true answer.

Notice that this time we have 2 types of activation functions, softmax and relu. ReLU (Rectified Linear Unit) is defined as:

**Insert RELU formula**

ReLU is regarded as the optimal activation function due to the quick and cheap derivate when using backpropagation. It is also a non-linear function, so while still quickly derivated and simple, it works quite well for the non-linearity needed on the network. As a general rule, this is the state-of-the-art activation function at the moment.

What about the last softmax function? Remember that our output is the probability of a given input to be classified as one of the 0-9 digits. Softmax function takes the intrinsict relationship between these classes to output a range of probabilities that ultimately add to 1. You can see the softmax function as a probability distribution of classes. This, obviously, is something we don't need in the input or hidden layers.

Notice also that we used `tensorflow.random_normal` function this time, as opposed to `tensorflow.zeros` function. When a network scales, we need to be careful about initialization. Having a network start at 0 creates a sort of bias, making it difficult for the network to scape local minima, thus not optimizing correctly, this is somewhat overcome by initializing the weights to some small random numbers.

### Optimization

Once we have defined our graph it is time to define the training process. Last time we talked about cross entropy function, this time we are going to do the same, but a little different, using TensorFlow's `softmax_cross_entropy_with_logits`:

```python
# Loss function
cost = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits(logits=y, labels=real_y))

# Optimiation
optimizer = tensorflow.train.AdamOptimizer(learning_rate=learning_rate)
train_step = optimizer.minimize(cost)
```

Notice we didn't use the `output` tensor for the loss function, this is due to the `softmax_cross_entropy_with_logits` function, which already calculates that, it only needs the "plain" output from the last layer.

This time we are going to use `AdamOptimizer`, long story short...it is considered the best optimizer at the moment. There is a lot of theory behind it, but just trust me this time. The learning rate was defined in the `parameters` section, you can easily change this to test how this affects the network, but you don't want it to be bigger than 1, but you do want it to be bigger than 0, for sure.

### Training

Last time we just ran the training one time for the whole training set, this was ok as it was a small example and a simple 1-layer MLP, but this time we need a more robust approach. We will have epochs, one epoch is equivalent to run all training data one time through the network, you can define a number of epochs before the training is stopped. There are other, more efficient ways of training, but let's go for this one.

```python
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
```

First, we initialize our variables with the `global_variables_initializer`, then we create a TensorFlow session and run the `init` operation.

For each epoch we are going to calculate how many samples we want by batch, we don't want to give the whole training set to the network at once. For each bath we grab the amount of samples we need and run the `train_step` over that data.

Now, we have trained define, the rest of the code is fairly similar to last time's example. After all the epochs we can use the MNIST `test` data set to calculate the networks accuracy.

Finally, we want to save this model, so we can use it on our UI data.

### Results

Last time our MLP was around `91%` of accuracy, since we are using a more complex network and training method (due to the implementation epochs), we are hoping to have a better accuracy percentage, and it delivers.

```python
Epoch 14
The final accuracy over the MNIST data is 94.34%
Saving model...
```

And this is just after 15 epochs, if you ran the script for 500 epochs we can actually achieve over `97%` accuracy. Be warned, 15 epochs took around 1 minute on my quad-core Core i7, while the 500 epochs took 30 minutes, big difference!

```python
Epoch 499
The final accuracy over the MNIST data is 97.84%
Saving model...
```

### Testing with the UI

If you got your `mnist_trained*` documents, then you just need to run `main.py` and start drawing some numbers.

**insert examples here**

Some numbers, even though are clear for you, they are not clear for the network, this could be due to the black and white grid we are using on the UI. The network was trained on grayscale images, so similar but yet different data sets.

# Conclusion

A more complex network achieved better results, but it also took more processing time. In my case, I can increase the performance by compiling TensorFlow for the specifics of my processor (at the moment there's a warning telling me that I don't have all the available enhancements on), or I could also use a GPU, but that's a story for another day.

If you want to download the complete code, get it at my [tensorflow-mnist](https://github.com/ctorresmx/tensorflow-mnist) repository.
