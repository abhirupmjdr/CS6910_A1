import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score,ConfusionMatrixDisplay
from keras.datasets import mnist, fashion_mnist
import matplotlib.pyplot as plt
import seaborn
import warnings
warnings.filterwarnings("ignore")


from keras.datasets import fashion_mnist
(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()


'''
The division by 255.0 is a normalization step for the input data.
In the context of image data, pixel values are often stored as 8-bit integers in 
the range 0 to 255. By dividing by 255, we scale these values to the range 0-1. 
This is a common preprocessing step for neural network inputs, 
as it can make the training process more stable and efficient.
'''

x_test = x_test / 255.0
x_train = x_train / 255.0

'''
Here at first we are splitting the training data into training and validation data
then, Reshaping the training, validation, and test input data and then transpose it
after this Reshaping the training, validation, and test output data

'''
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
x_train_T = x_train.reshape(-1, x_train.shape[1]*x_train.shape[2]).T
x_val_T = x_val.reshape(-1, x_val.shape[1]*x_val.shape[2]).T
x_test_T = x_test.reshape(-1, x_test.shape[1]*x_test.shape[2]).T
y_train_T, y_val_T, y_test_T = y_train.reshape(1, -1), y_val.reshape(1, -1), y_test.reshape(1, -1)

class Util:

    @staticmethod
    def apply_activation(A, activation):
        # Apply the specified activation function to the input which is the output of a layer in the neural network
            if activation == 'sigmoid':
                return Compute.sigmoid(A)
            elif activation == 'ReLU':
                return Compute.Relu(A)
            elif activation == 'tanh':
                return Compute.tanh(A)
            elif activation == 'identity':
                return Compute.indentiy(A)

    @staticmethod
    def loss(input, true_output, predicted_output, loss, batch_size,n_output):
        """
        Calculate the specified loss function between the true output and the predicted output.

        Parameters:
        input: The input to the neural network.
        true_output: The true output of the neural network.
        predicted_output: The output predicted by the neural network.
        loss: The name of the loss function to calculate.
        batch_size: The size of the batch for each iteration of training.
        n_output: The number of output units in the neural network.

        Returns:
        The calculated loss.
        """
        if loss == 'cross_entropy':
            one_hot_true_output = np.eye(n_output)[true_output[0]].T
            return -np.sum(one_hot_true_output * np.log(predicted_output + 1e-9)) / batch_size


        if loss=='mean_squared_error':
            one_hot_true_output = np.eye(n_output)[true_output[0]].T
            loss_factor=np.square(predicted_output-one_hot_true_output)
            return np.sum(loss_factor)/batch_size

    @staticmethod
    def accuracy(input, true_output, predicted_output):
        """
        Calculate the accuracy of the predicted output compared to the true output.

        Parameters:
        input: The input to the neural network.
        true_output: The true output of the neural network.
        predicted_output: The output predicted by the neural network.

        Returns:
        The calculated accuracy as a percentage.
        """
        predicted_labels = np.argmax(predicted_output, axis=0)
        correct_predictions = np.sum(true_output == predicted_labels)
        total_samples = true_output.shape[1]
        accuracy_percentage = (correct_predictions / total_samples) * 100
        return accuracy_percentage

'''
This class contains the different activation functions used in the neural network
The different activation functions used are:
1. Sigmoid
2. Softmax
3. ReLU
4. tanh
5. Identity

The class also contains the derivatives of the different activation functions, which are used in the backpropagation algorithm

'''
class Compute:

    @staticmethod
    def sigmoid(x):
        # sigmoid function Implementation used in hidden layers of the neural network
        return  1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        # softmax function Implementation which is used in the output layer of the neural network
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    @staticmethod
    def Relu(x):
        # relu function Implementation which is used in the hidden layers of the neural network
        return np.maximum(x,0)

    @staticmethod
    def tanh(x):
        # tanh function Implementation which is used in the hidden layers of the neural network
        return (2 * Compute.sigmoid(2 * x)) - 1
    
    @staticmethod
    def indentiy(x):
        # identity function Implementation which is used in the hidden layers of the neural network
        return x    

    @staticmethod
    def softmax_derivative(x):
        # softmax derivative function Implementation which is used in the backpropagation algorithm in the output layer of the neural network
        return Compute.softmax(x) * (1-Compute.softmax(x))

    @staticmethod
    def sigmoid_derivative(Z):
        # sigmoid derivative function Implementation which is used in the backpropagation algorithm in the hidden layers of the neural network
        s = Compute.sigmoid(Z)
        dA = s * (1 - s)
        return dA

    @staticmethod
    def Relu_derivative(x):
        # relu derivative function Implementation which is used in the backpropagation algorithm in the hidden layers of the neural network
        return 1*(x > 0)

    @staticmethod
    def tanh_derivative(x):
        # tanh derivative function Implementation which is used in the backpropagation algorithm in the hidden layers of the neural network
        return (1 - (Compute.tanh(x)**2))
    
    @staticmethod
    def identity_derivative(x):
        # identity derivative function Implementation which is used in the backpropagation algorithm in the hidden layers of the neural network
        return 1

    @staticmethod
    def calculate_gradients(k, dA, H_prev, A_prev, W, activation, batch_size):
        """
        Calculate the gradients for the weights and biases of the neural network for the current layer.
        This is used in the backpropagation algorithm.

        Parameters:
        k: The current layer number.
        dA: The derivative of the activation function.
        H_prev: The previous hidden state.
        A_prev: The previous activation.
        W: The weights of the current layer.
        activation: The activation function used in the current layer.
        batch_size: The size of the batch for each iteration of training.

        Returns:
        dW: The gradient of the weights.
        db: The gradient of the biases.
        dH_prev: The gradient of the previous hidden state.
        dA_prev: The gradient of the previous activation.

        """
        dW = Compute.calculate_dW(dA, H_prev, batch_size)
        db = Compute.calculate_db(dA, batch_size)
        dH_prev, dA_prev = Compute.calculate_dH_prev_dA_prev(k, W, dA, activation, A_prev) if k > 1 else (np.zeros(H_prev.shape), np.zeros(A_prev.shape))

        return dW, db, dH_prev, dA_prev
    @staticmethod
    def calculate_dW(dA, H_prev, batch_size):
            # Calculate the gradient of the weights used in the backpropagation algorithm
            return np.dot(dA, H_prev.T) / batch_size

    @staticmethod
    def calculate_db(dA, batch_size):
            # Calculate the gradient of the biases used in the backpropagation algorithm
            return np.sum(dA, axis=1, keepdims=True) / batch_size

    @staticmethod
    def calculate_dH_prev_dA_prev(k, W, dA, activation, A_prev):
            # Calculate the gradient of the previous hidden state and activation used in the backpropagation algorithm
            dH_prev = np.matmul(W.T, dA)
            dA_prev = Compute.calculate_dA_prev(dH_prev, activation, A_prev)
            return dH_prev, dA_prev

    @staticmethod
    def calculate_dA_prev(dH_prev, activation, A_prev):
            # Calculate the gradient of the previous activation used in the backpropagation algorithm
            if activation == 'sigmoid':
                return dH_prev * Compute.sigmoid_derivative(A_prev)
            elif activation == 'tanh':
                return dH_prev * Compute.tanh_derivative(A_prev)
            elif activation == 'ReLU':
                return dH_prev * Compute.Relu_derivative(A_prev)
            elif activation == 'identity':
                return dH_prev * Compute.identity_derivative(A_prev)

'''
This class contains the update rules for the different optimizers used to train the neural network
The different optimizers used are:
1. Stochastic Gradient Descent  
2. Nesterov Accelerated Gradient Descent 
3. Momentum Gradient Descent
4. RMSProp
5. Adam
6. Nadam

'''
class Update:
    @staticmethod
    def stochastic_gradient_descent(eta,theta,grads,n_layers,weight_decay=0):
        """
        Implements the Stochastic Gradient Descent (SGD) optimization algorithm for training a neural network.

        Parameters:
        eta: The learning rate.
        theta: The parameters of the neural network.
        grads: The gradients of the parameters.
        n_layers: The number of layers in the neural network.
        weight_decay: A regularization parameter (default is 0).

        """
        for l in range(1, n_layers):
            W, dW = theta["W" + str(l)], grads["dW" + str(l)]
            b, db = theta["b" + str(l)],grads["db" + str(l)]
            W -= eta * dW -eta*weight_decay*W
            b -= eta * db -eta*weight_decay*b
            theta["W" + str(l)], theta["b" + str(l)] = W, b
    # computing the theta for specifically nesterov accelerated gradient descent
    def compute_theta(my_network, mom, previous_updates):
        theta = {}
        for l in range(1, my_network.n_layers):
            theta["W" + str(l)] = my_network.theta["W" + str(l)] - mom * previous_updates["W" + str(l)]
            theta["b" + str(l)] = my_network.theta["b" + str(l)] - mom * previous_updates["b" + str(l)]
        return theta
    # computing the previous updates for specifically nesterov accelerated gradient descent
    def compute_previous_updates(my_network, mom, previous_updates):
        for l in range(1, my_network.n_layers):
            previous_updates["W" + str(l)] = mom * previous_updates["W" + str(l)] + (1-mom)*my_network.grads["dW" + str(l)]
            previous_updates["b" + str(l)] = mom * previous_updates["b" + str(l)] + (1-mom)*my_network.grads["db" + str(l)]
        return previous_updates
    
    #updating the weights and biases based on the gradients using the nesterov accelerated gradient descent algorithm
    def update_theta(my_network, eta, weight_decay):
        for l in range(1, my_network.n_layers):
            my_network.theta["W" + str(l)] -= eta * my_network.grads["dW" + str(l)] -eta*weight_decay*my_network.theta["W" + str(l)]
            my_network.theta["b" + str(l)] -= eta * my_network.grads["db" + str(l)] -eta*weight_decay*my_network.theta["b" + str(l)]

    def nesterov_gradient_descent(my_network, i, eta, batch_size, mom, previous_updates, loss, weight_decay=0):

        """
            Implements the Nesterov Accelerated Gradient Descent optimization algorithm for training a neural network.

            Parameters:
            my_network: The neural network to be trained.
            i: The current iteration.
            eta: The learning rate.
            batch_size: The size of the batch for each iteration of training.
            mom: The momentum factor.
            previous_updates: Dictionary to store the previous updates.
            loss: The loss function to be minimized.
            weight_decay: A regularization parameter (default is 0).

            Returns:
            previous_updates: Updated dictionary with the previous updates.

        """

        input_data = my_network.TrainInput[:, i:i + batch_size]
        output_data = my_network.TrainOutput[0, i:i + batch_size]
        
        theta = Update.compute_theta(my_network, mom, previous_updates)
        
        y_predicted = my_network.forward(input_data, my_network.activation_function, theta)
        e_y = np.transpose(np.eye(my_network.n_output)[output_data])
        
        my_network.backpropagation(y_predicted, e_y, batch_size, loss, my_network.activation_function, theta)
        
        previous_updates = Update.compute_previous_updates(my_network, mom, previous_updates)
        
        Update.update_theta(my_network, eta, weight_decay)
        
        return previous_updates

    @staticmethod
    def momentum_gradient_descent(my_network,eta, mom, previous_updates,weight_decay=0):

        """
        Implements the Momentum Gradient Descent optimization algorithm for training a neural network.

        Parameters:
        my_network: The neural network to be trained.
        eta: The learning rate.
        mom: The momentum factor.
        previous_updates: Dictionary to store the previous updates.
        weight_decay: A regularization parameter (default is 0).

        Returns:
        previous_updates: Updated dictionary with the previous updates.

        """

        for l in range(1, my_network.n_layers):
            uW, ub = previous_updates["W" + str(l)], previous_updates["b" + str(l)]
            W, dW = my_network.theta["W" + str(l)], my_network.grads["dW" + str(l)]
            b, db = my_network.theta["b" + str(l)], my_network.grads["db" + str(l)]
            uW = mom * uW + (1-mom) * dW
            ub = mom * ub + (1-mom) * db
            W -= eta * uW -eta*weight_decay*W
            b -= eta * ub  -eta*weight_decay*b
            previous_updates["W" + str(l)], previous_updates["b" + str(l)] = uW, ub
            my_network.theta["W" + str(l)], my_network.theta["b" + str(l)] = W, b
            return previous_updates

    @staticmethod
    def update_previous_updates(previous_updates, beta, grads, l, param):
        key = param + str(l)
        previous_updates[key] = beta * previous_updates[key] + (1 - beta) * np.square(grads["d" + key])
        return previous_updates
    
    def rms_prop(my_network,eta, beta, epsilon, previous_updates,weight_decay=0):

        """
            Implements the rmsprop optimization algorithm for training a neural network.

            Parameters:
            my_network: The neural network to be trained.
            eta: The learning rate.
            beta: The decay rate.
            epsilon: A small constant for numerical stability.
            previous_updates: Dictionary to store the previous updates.
            weight_decay: A regularization parameter (default is 0).

            Returns:
            previous_updates: Updated dictionary with the previous updates.

        """

        for l in range(1, my_network.n_layers):
            previous_updates = Update.update_previous_updates(previous_updates, beta, my_network.grads, l, "W")
            previous_updates = Update.update_previous_updates(previous_updates, beta, my_network.grads, l, "b")
            factorW = eta / (np.sqrt(previous_updates["W" + str(l)] + epsilon))
            factorb = eta / (np.sqrt(previous_updates["b" + str(l)] + epsilon))
            my_network.theta["W" + str(l)] -= factorW * my_network.grads["dW" + str(l)] - eta*weight_decay*my_network.theta["W" + str(l)]
            my_network.theta["b" + str(l)] -= factorb * my_network.grads["db" + str(l)] -eta*weight_decay*my_network.theta["b" + str(l)]
            return previous_updates
        
    # calculating the factors for specifically nadam optimizer
    def calculate_factors_nadam(eta, VW_corrected, Vb_corrected, epsilon):
        weight_factor = eta / (np.sqrt(VW_corrected) + epsilon)
        bias_factor = eta / (np.sqrt(Vb_corrected) + epsilon)
        return weight_factor, bias_factor
    
    # calculating the terms for specifically nadam optimizer
    def calculate_terms_nadam(beta1, t, l, grads):
        term1 = 1 - (beta1 ** t)
        weight_term = (1 - beta1) * grads["dW" + str(l)] / term1
        bias_term = (1 - beta1) * grads["db" + str(l)] / term1
        return weight_term, bias_term

    # updating theta for specifically nadam optimizer
    def update_theta_nadam(my_network, l, weight_factor, bias_factor, MW_corrected,Mb_corrected,beta1, weight_term, bias_term, eta, weight_decay):
        my_network.theta["W" + str(l)] -= weight_factor * (beta1 * MW_corrected + weight_term) - eta * weight_decay * my_network.theta["W" + str(l)]
        my_network.theta["b" + str(l)] -= bias_factor * (beta1 * Mb_corrected + bias_term) - eta * weight_decay * my_network.theta["b" + str(l)]

    @staticmethod
    def nadam(my_network,eta, beta1, beta2, epsilon, M, V, t,weight_decay=0):
        """
            Implements the nadam optimization algorithm for training a neural network.


            Parameters:
            my_network: The neural network to be trained.
            eta: The learning rate.
            beta1, beta2: Exponential decay rates for the moment estimates.
            epsilon: A small constant for numerical stability.
            M, V: Dictionaries to store the moving averages of the gradients and squared gradients respectively.
            t: The current timestep.
            weight_decay: A regularization parameter (default is 0).

            Returns:
            M, V: Updated dictionaries with moving averages of the gradients and squared gradients.

        """
        for l in range(1, my_network.n_layers):
            M["W" + str(l)] = beta1 * M["W" + str(l)] + (1 - beta1) * my_network.grads["dW" + str(l)]
            M["b" + str(l)] = beta1 * M["b" + str(l)] + (1 - beta1) * my_network.grads["db" + str(l)]
            MW_new = M["W" + str(l)] / (1 - (beta1 ** (t)))
            Mb_new = M["b" + str(l)] / (1 - (beta1 ** (t)))

            V["W" + str(l)] = beta2 * V["W" + str(l)] + (1 - beta2) * np.square(my_network.grads["dW" + str(l)])
            V["b" + str(l)] = beta2 * V["b" + str(l)] + (1 - beta2) * np.square(my_network.grads["db" + str(l)])
            VW_new = V["W" + str(l)] / (1 - (beta2 ** (t)))
            Vb_new = V["b" + str(l)] / (1 - (beta2 ** (t)))

            weight_factor, bias_factor = Update.calculate_factors_nadam(eta, VW_new, Vb_new, epsilon)
            weight_term, bias_term = Update.calculate_terms_nadam(beta1, t, l, my_network.grads)
            Update.update_theta_nadam(my_network, l, weight_factor, bias_factor, MW_new,Mb_new,beta1, weight_term, bias_term, eta, weight_decay)

        return M, V

    @staticmethod

    def adam(my_network,eta, beta1, beta2, epsilon, M, V, t,weight_decay=0): #taken from slide-2 page 42 [cs6910]
        """
            Implements the Adam optimization algorithm for training a neural network.

            Input Parameters:
            my_network: The neural network to be trained.
            eta: The learning rate.
            beta1, beta2: Exponential decay rates for the moment estimates.
            epsilon: A small constant for numerical stability.
            M, V: Dictionaries to store the moving averages of the gradients and squared gradients respectively.
            t: The current timestep.
            weight_decay: A regularization parameter (default is 0).

            Returns:
            M, V: Updated dictionaries with moving averages of the gradients and squared gradients.

        """
        for l in range(1, my_network.n_layers):
            M["W" + str(l)] = beta1 * M["W" + str(l)] + (1 - beta1) * my_network.grads["dW" + str(l)]
            M["b" + str(l)] = beta1 * M["b" + str(l)] + (1 - beta1) * my_network.grads["db" + str(l)]
            V["W" + str(l)] = beta2 * V["W" + str(l)] + (1 - beta2) * np.square(my_network.grads["dW" + str(l)])
            V["b" + str(l)] = beta2 * V["b" + str(l)] + (1 - beta2) * np.square(my_network.grads["db" + str(l)])
            MW_hat = M["W" + str(l)] / (1 - np.power(beta1,t))
            Mb_hat = M["b" + str(l)] / (1 - np.power(beta1,t))
            VW_hat = V["W" + str(l)] / (1 - np.power(beta2,t))
            Vb_hat = V["b" + str(l)] / (1 - np.power(beta2,t))
            my_network.theta["W" + str(l)] -= (eta / (np.sqrt(VW_hat) + epsilon)) * MW_hat -eta*weight_decay*my_network.theta["W" + str(l)]
            my_network.theta["b" + str(l)] -= (eta / (np.sqrt(Vb_hat) + epsilon)) * Mb_hat -eta*weight_decay*my_network.theta["b" + str(l)]
        return M, V

class MyNeuralNetwork:
  mode_of_initialization = ""
  n_layers = 0
  activation_function = ""
  n_input = 0
  n_output = 0
  n_neurons = []
  TrainInput = []
  TrainOutput = []
  ValInput = []
  ValOutput = []
  theta = {}
  cache = {}
  grads = {}
  def initialize_weights(self, l):
    if self.mode_of_initialization == "random":
        # Calculate the shape of the weight matrix
        shape = (self.n_neurons[l], self.n_neurons[l - 1])
        # Generate the weight matrix with random values
        self.theta[f"W{l}"] = np.random.randn(*shape)
    elif self.mode_of_initialization == "Xavier":
        # Calculate the sum of the number of neurons in the current layer and the previous layer
        neuron_sum = self.n_neurons[l - 1] + self.n_neurons[l]

        # Convert the sum to a float
        neuron_sum_float = float(neuron_sum)

        # Calculate the divisor for the square root operation
        divisor = 2 / neuron_sum_float

        # Calculate the limit for the Xavier initialization
        limit = np.sqrt(divisor)

        # Generate a matrix of random values from the normal distribution
        # The mean of the distribution is 0.0 and the standard deviation is equal to the limit
        random_values = np.random.normal(0.0, limit, size=(self.n_neurons[l], self.n_neurons[l - 1]))

        # Assign the matrix of random values to the weights of the l-th layer
        self.theta["W" + str(l)] = random_values

  def initialize_biases(self, l):
    self.theta["b" + str(l)] = np.zeros((self.n_neurons[l] , 1))

  def setup_layers(self, number_of_hidden_layers, num_neurons_in_hidden_layers):
    neuronsPerLayer = [num_neurons_in_hidden_layers] * number_of_hidden_layers
    self.n_layers = number_of_hidden_layers + 2
    self.n_neurons = neuronsPerLayer
    self.n_neurons.append(self.n_output)
    self.n_neurons.insert(0 , self.n_input)


  def __init__(self,mode_of_initialization="random",number_of_hidden_layers=1,num_neurons_in_hidden_layers=4,activation="sigmoid",TrainInput=x_train_T,TrainOutput=y_train_T,ValInput=x_val_T,ValOutput=y_val_T):
    self.mode_of_initialization = mode_of_initialization
    self.activation_function = activation
    self.TrainInput = TrainInput
    self.TrainOutput = TrainOutput
    self.n_input = TrainInput.shape[0]
    # Find the index of the maximum value in each row of TrainOutput
    max_index_in_each_row = TrainOutput.argmax(axis = 1)
    # Select the first of these indices
    first_max_index = max_index_in_each_row[0]
    # Use this index to index into TrainOutput
    value_at_max_index = TrainOutput[0, first_max_index]

    self.n_output = value_at_max_index + 1
    self.cache["H0"] = TrainInput
    self.cache["A0"] = TrainInput
    self.grads = {}
    self.ValInput = ValInput
    self.ValOutput = ValOutput
    self.setup_layers(number_of_hidden_layers, num_neurons_in_hidden_layers)
    for l in range(1,self.n_layers):
        self.initialize_weights( l)
        self.initialize_biases( l)

    
  def forward(self, X, activation, theta):
    self.cache["H0"] = X
    for l in range(1, self.n_layers):
        self.cache["A" + str(l)] = np.dot(self.theta["W" + str(l)], self.cache["H" + str(l - 1)]) + self.theta["b" + str(l)]
        A=self.cache["A" + str(l)]
        H = Util.apply_activation(A, activation)
        self.cache["H" + str(l)] = H
    y_hat= Compute.softmax(self.cache["A" + str(self.n_layers - 1)])

    return y_hat

  def backpropagation(self, y_predicted, e_y, batch_size, loss, activation, theta):
        if loss == 'cross_entropy':
            dA = y_predicted - e_y
        elif loss=='mean_squared_error':
            dA=(y_predicted - e_y)*Compute.softmax_derivative(self.cache["A" + str(self.n_layers - 1)])
        self.grads["dA" + str(self.n_layers - 1)] = dA

        for k in range(self.n_layers - 1, 0, -1):
            dA, H_prev, A_prev, W =self.grads["dA" + str(k)],self.cache["H" + str(k - 1)],self.cache["A" + str(k - 1)],self.theta["W" + str(k)]
            dW, db, dH_prev, dA_prev = Compute.calculate_gradients(k, dA, H_prev, A_prev, W, activation, batch_size)

            self.grads["dA" + str(k - 1)] = dA_prev
            self.grads["dW" + str(k)] = dW
            self.grads["db" + str(k)] = db

        return




  def compute(self, eta = 0.1,mom=0.5,beta = 0.5,beta1 = 0.5,beta2 = 0.5 ,epsilon = 0.000001, optimizer = 'sgd',batch_size = 4,weight_decay=0,loss = 'cross_entropy',epochs = 1):
    train_c_epoch, tarin_acc_per_epoch, val_c_per_epoch, val_acc_per_epoch, previous_updates, M, V = [], [], [], [], {}, {}, {}
    for l in range(1 , self.n_layers):
      previous_updates["W" + str(l)] = np.zeros((self.n_neurons[l] , self.n_neurons[l - 1]))
      previous_updates["b" + str(l)] = np.zeros((self.n_neurons[l] , 1))
      M["W" + str(l)] = np.zeros((self.n_neurons[l] , self.n_neurons[l - 1]))
      M["b" + str(l)] = np.zeros((self.n_neurons[l] , 1))
      V["W" + str(l)] = np.zeros((self.n_neurons[l] , self.n_neurons[l - 1]))
      V["b" + str(l)] = np.zeros((self.n_neurons[l] , 1))
    t = 1
    for count in range(epochs):
      for i in range(0 , self.TrainInput.shape[1],batch_size):
        if i + batch_size > self.TrainInput.shape[1]:
          continue
        theta = self.theta
        train_batch_input = self.TrainInput[:,i:i + batch_size]
        train_batch_output = self.TrainOutput[0,i : i + batch_size]
        yPredicted = self.forward(train_batch_input,self.activation_function,theta)
        e_y = np.transpose(np.eye(self.n_output)[train_batch_output])
        self.backpropagation(yPredicted,e_y,batch_size,loss,self.activation_function,theta)
        if optimizer == 'sgd':   #referred slide page 54
            Update.stochastic_gradient_descent(eta,self.theta,self.grads,self.n_layers,weight_decay) #working
        elif optimizer == 'nag':
            previous_updates=Update.nesterov_gradient_descent(self,i,eta, batch_size, mom, previous_updates,loss,weight_decay) #working

        elif optimizer == 'momentum': #referred from slide 43
          previous_updates=Update.momentum_gradient_descent(self,eta,mom,previous_updates,weight_decay) #working

        elif optimizer == 'rmsprop':
          previous_updates=Update.rms_prop(self,eta,beta,epsilon,previous_updates,weight_decay) #working
        elif optimizer == 'adam':

          M , V = Update.adam(self,eta,beta1,beta2,epsilon,M , V , count+1,weight_decay)
        elif optimizer == 'nadam':

          M , V = Update.nadam(self,eta,beta1,beta2,epsilon,M , V , count+1,weight_decay)

      y_hat = self.forward(self.TrainInput,self.activation_function,self.theta)
      valy_hat = self.forward(self.ValInput,self.activation_function,self.theta)
      train_cost = Util.loss(self.TrainInput,self.TrainOutput,y_hat,loss,self.TrainInput.shape[1],self.n_output)
      train_c_epoch.append(train_cost)
      val_cost = Util.loss(self.ValInput,self.ValOutput,valy_hat,loss,self.ValInput.shape[1],self.n_output)
      val_c_per_epoch.append(val_cost)
      tarin_acc_per_epoch.append(Util.accuracy(self.TrainInput, self.TrainOutput,y_hat))
      val_acc_per_epoch.append(Util.accuracy(self.ValInput, self.ValOutput,valy_hat))


      print("---------"*20)
      print(f"Epoch  = {format(count+1)}")
      print(f"Training Accuracy = {format(tarin_acc_per_epoch[-1])}")
      print(f"Validation Accuracy = {format(val_acc_per_epoch[-1])}")
    return train_c_epoch,tarin_acc_per_epoch,val_c_per_epoch,val_acc_per_epoch



my_network = MyNeuralNetwork(mode_of_initialization="Xavier",number_of_hidden_layers=3,num_neurons_in_hidden_layers=128,activation="tanh",TrainInput=x_train_T,TrainOutput=y_train_T,ValInput=x_val_T,ValOutput=y_val_T)
train=my_network.compute(eta = 0.0001,mom=0.5,beta = 0.9,beta1 = 0.9,beta2 = 0.9,epsilon =1e-9, optimizer = 'nag',batch_size = 32,weight_decay=0.5,loss = 'cross_entropy',epochs = 5)
 
'''
    This is simple backprop code with different optimizers and activation functions.
    so we have not to use anything related to wandb 
'''

