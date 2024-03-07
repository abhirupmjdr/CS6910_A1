
#question 2

import numpy as np
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split

# Load Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize the pixel values
x_test = x_test / 255.0
x_train = x_train / 255.0

# Split the training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# Reshape the data for the neural network
x_val_T = np.transpose(x_val.reshape(x_val.shape[0], -1))
x_train_T = np.transpose(x_train.reshape(x_train.shape[0], -1))
x_test_T = np.transpose(x_test.reshape(x_test.shape[0], -1))
y_train_T = y_train.reshape(1, -1)
y_val_T = y_val.reshape(1, -1)
y_test_T = y_test.reshape(1, -1)

class NeuralNetwork:
    def __init__(self, init_mode='random', num_hidden_layers=3, neurons_per_hidden_layer=64,
                 activation_function='sigmoid', train_input=x_train_T, train_output=y_train_T, val_input=x_val_T,
                 val_output=y_val_T):
        self.init_mode = init_mode
        self.num_hidden_layers = num_hidden_layers
        self.neurons_per_hidden_layer = neurons_per_hidden_layer
        self.activation_function = activation_function
        self.train_input = train_input
        self.train_output = train_output
        self.val_input = val_input
        self.val_output = val_output
        self.n_layers = num_hidden_layers + 2
        self.n_input = train_input.shape[0]
        self.n_output = np.max(train_output) + 1
        self.n_neurons = [self.n_input] + [neurons_per_hidden_layer] * num_hidden_layers + [self.n_output]
        self.cache = {"H0": train_input, "A0": train_input}
        self.theta = {}
        self.grads = {}
        self.initialize_parameters()

    def initialize_parameters(self):
        for l in range(1, self.n_layers):
            if self.init_mode == "random":
                self.theta[f"W{l}"] = np.random.randn(self.n_neurons[l], self.n_neurons[l - 1])
            self.theta[f"b{l}"] = np.zeros((self.n_neurons[l], 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(x, 0)

    def tanh(self, x):
        return np.tanh(x)

    def feedforward(self, input_data):
        H = input_data
        activation = self.activation_function
        for layer in range(1, self.n_layers - 1):
            weights = self.theta[f"W{layer}"]
            bias = self.theta[f"b{layer}"]
            activation_input = np.dot(weights, H) + bias
            H = getattr(self, activation)(activation_input)
        weights = self.theta[f"W{self.n_layers - 1}"]
        bias = self.theta[f"b{self.n_layers - 1}"]
        activation_input = np.dot(weights, H) + bias
        softmax_output = np.exp(activation_input) / np.sum(np.exp(activation_input), axis=0)
        return softmax_output

# Create an instance of the NeuralNetwork class
my_model = NeuralNetwork()

# Perform a feedforward pass on the training data
my_model.feedforward(x_train_T)
