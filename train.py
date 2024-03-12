
import wandb
from wandb.keras import WandbCallback
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score,ConfusionMatrixDisplay
from keras.datasets import mnist, fashion_mnist
import matplotlib.pyplot as plt
import seaborn
import warnings
warnings.filterwarnings("ignore")

wandb.login()
project_name = "cs6910-assignment1"
entity_name = "cs23m006"

import argparse
parser = argparse.ArgumentParser(description='Argument Parser for my neural network train.py file.')

parser.add_argument('-wp', '--wandb_project', default='cs6910-a1', help='Project name used to track experiments in Weights & Biases dashboard')
parser.add_argument('-we', '--wandb_entity', default='abhirupmjdr_dl-org', help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')
parser.add_argument('-d', '--dataset', default='fashion_mnist', choices=["mnist", "fashion_mnist"], help='Dataset to be used for training.')
parser.add_argument('-e', '--epochs', type=int, default=1, help='Number of epochs to train neural network.')
parser.add_argument('-b', '--batch_size', type=int, default=4, help='Batch size used to train neural network.')
parser.add_argument('-l', '--loss', default='cross_entropy', choices=["mean_squared_error", "cross_entropy"], help='Loss function to be used.')
parser.add_argument('-o', '--optimizer', default='sgd', choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help='Optimizer algorithm.')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.1, help='Learning rate used to optimize model parameters')
parser.add_argument('-m', '--momentum', type=float, default=0.5, help='Momentum used by momentum and nag optimizers.')
parser.add_argument('-beta', '--beta', type=float, default=0.5, help='Beta used by rmsprop optimizer')
parser.add_argument('-beta1', '--beta1', type=float, default=0.5, help='Beta1 used by adam and nadam optimizers.')
parser.add_argument('-beta2', '--beta2', type=float, default=0.5, help='Beta2 used by adam and nadam optimizers.')
parser.add_argument('-eps', '--epsilon', type=float, default=0.000001, help='Epsilon used by optimizers.')
parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0, help='Weight decay used by optimizers.')
parser.add_argument('-w_i', '--weight_init', default='random', choices=["random", "Xavier"], help='Weight initialization method.')
parser.add_argument('-nhl', '--num_layers', type=int, default=1, help='Number of hidden layers used in feedforward neural network.')
parser.add_argument('-sz', '--hidden_size', type=int, default=4, help='Number of hidden neurons in a feedforward layer.')
parser.add_argument('-a', '--activation', default='sigmoid', choices=["identity", "sigmoid", "tanh", "ReLU"], help='Activation function.')

args = parser.parse_args()

if(args.wandb_entity):  
    entity_name = args.wandb_entity
if(args.wandb_project):
    project_name = args.wandb_project

wandb.init(project=project_name, entity=entity_name)

from keras.datasets import fashion_mnist
(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

if(args.dataset == 'mnist'):
    (x_train,y_train),(x_test,y_test) = mnist.load_data()


x_test = x_test / 255.0
x_train = x_train / 255.0


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
x_train_T = x_train.reshape(-1, x_train.shape[1]*x_train.shape[2]).T
x_val_T = x_val.reshape(-1, x_val.shape[1]*x_val.shape[2]).T
x_test_T = x_test.reshape(-1, x_test.shape[1]*x_test.shape[2]).T
y_train_T, y_val_T, y_test_T = y_train.reshape(1, -1), y_val.reshape(1, -1), y_test.reshape(1, -1)

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


class Update:
    @staticmethod
    def stochastic_gradient_descent(eta,theta,grads,n_layers,weight_decay=0):
        for l in range(1, n_layers):
            W, dW = theta["W" + str(l)], grads["dW" + str(l)]
            b, db = theta["b" + str(l)],grads["db" + str(l)]
            W -= eta * dW -eta*weight_decay*W
            b -= eta * db -eta*weight_decay*b
            theta["W" + str(l)], theta["b" + str(l)] = W, b

    def compute_theta(my_network, mom, previous_updates):
        theta = {}
        for l in range(1, my_network.n_layers):
            theta["W" + str(l)] = my_network.theta["W" + str(l)] - mom * previous_updates["W" + str(l)]
            theta["b" + str(l)] = my_network.theta["b" + str(l)] - mom * previous_updates["b" + str(l)]
        return theta

    def compute_previous_updates(my_network, mom, previous_updates):
        for l in range(1, my_network.n_layers):
            previous_updates["W" + str(l)] = mom * previous_updates["W" + str(l)] + (1-mom)*my_network.grads["dW" + str(l)]
            previous_updates["b" + str(l)] = mom * previous_updates["b" + str(l)] + (1-mom)*my_network.grads["db" + str(l)]
        return previous_updates

    def update_theta(my_network, eta, weight_decay):
        for l in range(1, my_network.n_layers):
            my_network.theta["W" + str(l)] -= eta * my_network.grads["dW" + str(l)] -eta*weight_decay*my_network.theta["W" + str(l)]
            my_network.theta["b" + str(l)] -= eta * my_network.grads["db" + str(l)] -eta*weight_decay*my_network.theta["b" + str(l)]

    def nesterov_gradient_descent(my_network, i, eta, batch_size, mom, previous_updates, loss, weight_decay=0):
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
        for l in range(1, my_network.n_layers):
            previous_updates = Update.update_previous_updates(previous_updates, beta, my_network.grads, l, "W")
            previous_updates = Update.update_previous_updates(previous_updates, beta, my_network.grads, l, "b")
            factorW = eta / (np.sqrt(previous_updates["W" + str(l)] + epsilon))
            factorb = eta / (np.sqrt(previous_updates["b" + str(l)] + epsilon))
            my_network.theta["W" + str(l)] -= factorW * my_network.grads["dW" + str(l)] - eta*weight_decay*my_network.theta["W" + str(l)]
            my_network.theta["b" + str(l)] -= factorb * my_network.grads["db" + str(l)] -eta*weight_decay*my_network.theta["b" + str(l)]
            return previous_updates

    def calculate_factors_nadam(eta, VW_corrected, Vb_corrected, epsilon):
        weight_factor = eta / (np.sqrt(VW_corrected) + epsilon)
        bias_factor = eta / (np.sqrt(Vb_corrected) + epsilon)
        return weight_factor, bias_factor

    def calculate_terms_nadam(beta1, t, l, grads):
        term1 = 1 - (beta1 ** t)
        weight_term = (1 - beta1) * grads["dW" + str(l)] / term1
        bias_term = (1 - beta1) * grads["db" + str(l)] / term1
        return weight_term, bias_term

    def update_theta_nadam(my_network, l, weight_factor, bias_factor, MW_corrected,Mb_corrected,beta1, weight_term, bias_term, eta, weight_decay):
        my_network.theta["W" + str(l)] -= weight_factor * (beta1 * MW_corrected + weight_term) - eta * weight_decay * my_network.theta["W" + str(l)]
        my_network.theta["b" + str(l)] -= bias_factor * (beta1 * Mb_corrected + bias_term) - eta * weight_decay * my_network.theta["b" + str(l)]

    @staticmethod
    def nadam(my_network,eta, beta1, beta2, epsilon, M, V, t,weight_decay=0):
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
          epsilon = 1e-10
          M , V = Update.adam(self,eta,beta1,beta2,epsilon,M , V , count+1,weight_decay)
        elif optimizer == 'nadam':
          epsilon = 1e-8
          M , V = Update.nadam(self,eta,beta1,beta2,epsilon,M , V , count+1,weight_decay)

      y_hat = self.forward(self.TrainInput,self.activation_function,self.theta)
      valy_hat = self.forward(self.ValInput,self.activation_function,self.theta)
      train_cost = Util.loss(self.TrainInput,self.TrainOutput,y_hat,loss,self.TrainInput.shape[1],self.n_output)
      train_c_epoch.append(train_cost)
      val_cost = Util.loss(self.ValInput,self.ValOutput,valy_hat,loss,self.ValInput.shape[1],self.n_output)
      val_c_per_epoch.append(val_cost)
      tarin_acc_per_epoch.append(Util.accuracy(self.TrainInput, self.TrainOutput,y_hat))
      val_acc_per_epoch.append(Util.accuracy(self.ValInput, self.ValOutput,valy_hat))
    #   print(np.eye(self.n_output)[self.ValOutput[0]].T.shape,valy_hat.shape)

      print("---------"*20)
      print(f"Epoch  = {format(count+1)}")
      print(f"Training Accuracy = {format(tarin_acc_per_epoch[-1])}")
      print(f"Validation Accuracy = {format(val_acc_per_epoch[-1])}")
    #   if(count==1):
    #     print(self.cache["A1"])
      wandb.log({"training_accuracy": tarin_acc_per_epoch[-1],"validation_accuracy": val_acc_per_epoch[-1],"training_loss":train_cost,"validation_loss": val_cost,"epoch": count})
    return train_c_epoch,tarin_acc_per_epoch,val_c_per_epoch,val_acc_per_epoch


my_network = MyNeuralNetwork(mode_of_initialization=args.weight_init,number_of_hidden_layers=args.num_layers,num_neurons_in_hidden_layers=args.hidden_size,activation=args.activation,TrainInput=x_train_T,TrainOutput=y_train_T,ValInput=x_val_T,ValOutput=y_val_T)
train=my_network.compute(eta = args.learning_rate,mom=args.momentum,beta = args.beta,beta1 = args.beta1,beta2 = args.beta2 ,epsilon = args.epsilon, optimizer = args.optimizer,batch_size =args.batch_size,weight_decay=args.weight_decay,loss = args.loss,epochs = args.epochs)


