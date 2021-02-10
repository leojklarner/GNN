import numpy as np


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise  NotImplementedError

    def backward(self, output_error, learning_rate):
        raise NotImplementedError


class FCLayer():

    def __init__(self, input_size, output_size, activation, activation_derivative):
        self.input = None
        self.output = None
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(1, output_size)

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(np.dot(self.input, self.weights) + self.bias)
        return self.output

    def backward(self, output_error, learning_rate):
        input_error = np.multiply(np.dot(output_error, self.weights.T), self.activation_derivative(self.input))
        weights_error = np.dot(self.input.T, output_error)
        bias_error = output_error

        # update parameters
        self.weights -= learning_rate*weights_error
        self.bias -= learning_rate*bias_error

        return input_error


