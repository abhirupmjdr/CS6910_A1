
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

class Util:

    @staticmethod
    def apply_activation(A, activation):
            if activation == 'sigmoid':
                return Compute.sigmoid(A)
            elif activation == 'ReLU':
                return Compute.Relu(A)
            elif activation == 'tanh':
                return Compute.tanh(A)

    @staticmethod
    def loss(input, true_output, predicted_output, loss, batch_size,n_output):
        if loss == 'cross_entropy':
            one_hot_true_output = np.eye(n_output)[true_output[0]].T
            return -np.sum(one_hot_true_output * np.log(predicted_output + 1e-9)) / batch_size


        if loss=='mean_squared_error':
            one_hot_true_output = np.eye(n_output)[true_output[0]].T
            loss_factor=np.square(predicted_output-one_hot_true_output)
            return np.sum(loss_factor)/batch_size

    @staticmethod
    def accuracy(input, true_output, predicted_output):
        predicted_labels = np.argmax(predicted_output, axis=0)
        correct_predictions = np.sum(true_output == predicted_labels)
        total_samples = true_output.shape[1]
        accuracy_percentage = (correct_predictions / total_samples) * 100
        return accuracy_percentage


class Compute:

    @staticmethod
    def sigmoid(x):
        return  1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    @staticmethod
    def Relu(x):
        return np.maximum(x,0)

    @staticmethod
    def tanh(x):
        return (2 * Compute.sigmoid(2 * x)) - 1

    @staticmethod
    def softmax_derivative(x):
        return Compute.softmax(x) * (1-Compute.softmax(x))

    @staticmethod
    def sigmoid_derivative(Z):
        s = Compute.sigmoid(Z)
        dA = s * (1 - s)
        return dA

    @staticmethod
    def Relu_derivative(x):
        return 1*(x > 0)

    @staticmethod
    def tanh_derivative(x):
        return (1 - (Compute.tanh(x)**2))

    @staticmethod
    def calculate_gradients(k, dA, H_prev, A_prev, W, activation, batch_size):
            dW = Compute.calculate_dW(dA, H_prev, batch_size)
            db = Compute.calculate_db(dA, batch_size)
            dH_prev, dA_prev = Compute.calculate_dH_prev_dA_prev(k, W, dA, activation, A_prev) if k > 1 else (np.zeros(H_prev.shape), np.zeros(A_prev.shape))

            return dW, db, dH_prev, dA_prev
    @staticmethod
    def calculate_dW(dA, H_prev, batch_size):
            return np.dot(dA, H_prev.T) / batch_size

    @staticmethod
    def calculate_db(dA, batch_size):
            return np.sum(dA, axis=1, keepdims=True) / batch_size

    @staticmethod
    def calculate_dH_prev_dA_prev(k, W, dA, activation, A_prev):
            dH_prev = np.matmul(W.T, dA)
            dA_prev = Compute.calculate_dA_prev(dH_prev, activation, A_prev)
            return dH_prev, dA_prev

    @staticmethod
    def calculate_dA_prev(dH_prev, activation, A_prev):
            if activation == 'sigmoid':
                return dH_prev * Compute.sigmoid_derivative(A_prev)
            elif activation == 'tanh':
                return dH_prev * Compute.tanh_derivative(A_prev)
            elif activation == 'ReLU':
                return dH_prev * Compute.Relu_derivative(A_prev)


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
            elif self.init_mode == "Xavier":
                limit = np.sqrt(2 / float(self.n_neurons[l - 1] + self.n_neurons[l]))
                self.theta["W" + str(l)] = np.random.normal(0.0, limit, size=(self.n_neurons[l],self.n_neurons[l - 1]))
            self.theta["b" + str(l)] = np.zeros((self.n_neurons[l] , 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(x, 0)

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, X):
        self.cache["H0"] = X
        for l in range(1, self.n_layers):
            H = self.cache["H" + str(l - 1)]
            W = self.theta["W" + str(l)]
            b = self.theta["b" + str(l)]
            A = np.dot(W, H) + b
            self.cache["A" + str(l)] = A
            H = Util.apply_activation(A, self.activation_function)
            self.cache["H" + str(l)] = H
        Al = self.cache["A" + str(self.n_layers - 1)]
        y_hat= Compute.softmax(Al)
        return y_hat

# Create an instance of the NeuralNetwork class
my_model = NeuralNetwork()

# Perform a feedforward pass on the training data
my_model.forward(x_train_T)
