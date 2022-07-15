import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        self.size = len(layer_sizes) - 1
        self.weights = []
        self.biases = []

        for i in range(1, len(layer_sizes)):
            W = np.random.normal(0, .5, size=(layer_sizes[i], layer_sizes[i-1]))
            b = np.zeros((layer_sizes[i]))
            self.weights.append(W)
            self.biases.append(b)

        pass

    def activation(self, x):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        return 1 / (1 + np.exp(-x))
        pass

    def forward(self, x):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """

        res = x
        for i in range(self.size):
            res = self.activation(np.matmul(self.weights[i], res) + self.biases[i])

        return res
