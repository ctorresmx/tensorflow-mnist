#!/usr/bin/env python

import numpy
import tkinter as tk
import tensorflow
import logging


class App(tk.Tk):
    def __init__(self, *args, **kwargs):
        # Creates the 28x28 grid for MNIST data.
        self.rows = kwargs.pop('rows')
        self.cols = kwargs.pop('cols')
        self.cell_width = kwargs.pop('cell_width', 10)
        self.cell_height = kwargs.pop('cell_height', 10)
        self.width = self.cols * self.cell_width
        self.height = self.rows * self.cell_height
        self.grid = numpy.zeros([28, 28])

        # Creates the tk window and canvas
        tk.Tk.__init__(self, *args, **kwargs)
        self.title('Draw your digit')
        self.canvas = tk.Canvas(self, width=self.width, height=self.height,
                                borderwidth=0, highlightthickness=0)
        self.canvas.pack(side="top", fill="both", expand="true")

        # Draws the basic UI
        self.rect = {}
        self.oval = {}
        for column in range(self.rows):
            for row in range(self.cols):
                x1 = column * self.cell_width
                y1 = row * self.cell_height
                x2 = x1 + self.cell_width
                y2 = y1 + self.cell_height
                self.rect[row, column] = self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill='white', tags="rect"
                )

        # Adds the event handlers
        self.bind('<B1-Motion>', self.motion)
        self.bind('<ButtonRelease-1>', self.predict)
        self.resizable(0, 0)

        # Loads the TensorFlow model
        self.load_model('mnist_trained.meta')

        # Creates the application logger
        self.configure_logger()

        # Creates the additional prediction window
        self.prediction_window = PredictionWindow()

        # Makes <ESC> key the close app event
        self.bind('<Escape>', lambda x: self.destroy())

    def destroy(self):
        self.prediction_window.destroy()
        super().destroy()

    def configure_logger(self):
        """ Helper function to create a default logger. """
        self.logger = logging.getLogger()

        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)-8s %(message)s')
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)

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

    def predict(self, event):
        """ Predict digit drawn on the grid, then clear the grid. """

        # Feed the grid as a 1-dimensional array to the TensorFlow model.
        feed_dict = {self.input: self.grid.reshape([1, 784])}

        # The output is a one-hot encoded array, we need to transform that array
        # of probabilities into a single digit, the one with the highest
        # probability.
        output = self.tensorflow_session.run(self.predict, feed_dict)
        predicted_digit = numpy.argmax(output)
        self.logger.info('Predicted digit: {}'.format(predicted_digit))
        self.prediction_window.prediction_label.config(text=predicted_digit)

        # Clear the grid, both, visually and in the array.
        for column in range(self.rows):
            for row in range(self.cols):
                x1 = column * self.cell_width
                y1 = row * self.cell_height
                x2 = x1 + self.cell_width
                y2 = y1 + self.cell_height
                self.rect[row, column] = self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill='white', tags="rect"
                )
        self.grid = numpy.zeros([28, 28])

    def motion(self, event):
        """ Mouse movement event handler. """
        row, col = event.y // 10, event.x // 10
        self.logger.info('Filling cell {}, {}'.format(row, col))

        x1 = col * self.cell_width
        y1 = row * self.cell_height
        x2 = x1 + self.cell_width
        y2 = y1 + self.cell_height

        self.rect[row, col] = self.canvas.create_rectangle(
            x1, y1, x2, y2, fill='black', tags="rect"
        )

        self.grid[row, col] = 1


class PredictionWindow(tk.Tk):
    def __init__(self, *args, **kwargs):
        # Creates the tk prediction window
        tk.Tk.__init__(self, *args, **kwargs)
        self.title('Digit predicted')

        self.prediction_label = tk.Label(self, text='', width=1, font=("Courier", 64))
        self.prediction_label.pack()


def main():
    app = App(rows=28, cols=28)
    app.mainloop()


if __name__ == "__main__":
    main()
