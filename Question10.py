
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


from keras.datasets import fashion_mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()



x_test = x_test / 255.0
x_train = x_train / 255.0

# Split the training data into training and validation sets
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
            elif activation == 'identity':
                return Compute.identity(A)

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
    def identity(x):
        return x    

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
    def identity_derivative(x):
        return 1    

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
            elif activation == 'identity': 
                return dH_prev * Compute.identity_derivative(A_prev)


class Update:
    @staticmethod
    def stochastic_gradient_descent(eta,theta,grads,n_layers,weight_decay=0):
        for l in range(1, n_layers):
            W, dW = theta["W" + str(l)], grads["dW" + str(l)]
            b, db = theta["b" + str(l)],grads["db" + str(l)]
            W -= eta * dW -eta*weight_decay*W
            b -= eta * db -eta*weight_decay*b
            theta["W" + str(l)], theta["b" + str(l)] = W, b

    @staticmethod
    def nesterov_gradient_descent(my_network,i,eta, batch_size, mom, previous_updates,loss,weight_decay=0):
        theta = {}
        input_data = my_network.TrainInput[:, i:i + batch_size]
        output_data = my_network.TrainOutput[0, i:i + batch_size]
        for l in range(1, my_network.n_layers):
            theta["W" + str(l)] = my_network.theta["W" + str(l)] - mom * previous_updates["W" + str(l)]
            theta["b" + str(l)] = my_network.theta["b" + str(l)] - mom * previous_updates["b" + str(l)]
        y_predicted = my_network.forward(input_data, my_network.activation_function, my_network.theta)
        e_y = np.transpose(np.eye(my_network.n_output)[output_data])
        my_network.backpropagation(y_predicted, e_y, batch_size, loss, my_network.activation_function, my_network.theta)
        for l in range(1, my_network.n_layers):
            previous_updates["W" + str(l)] = mom * previous_updates["W" + str(l)] + (1-mom)*my_network.grads["dW" + str(l)]
            previous_updates["b" + str(l)] = mom * previous_updates["b" + str(l)] + (1-mom)*my_network.grads["db" + str(l)]
            my_network.theta["W" + str(l)] -= eta * my_network.grads["dW" + str(l)] -eta*weight_decay*my_network.theta["W" + str(l)]
            my_network.theta["b" + str(l)] -= eta * my_network.grads["db" + str(l)] -eta*weight_decay*my_network.theta["b" + str(l)]
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
    def rms_prop(my_network,eta, beta, epsilon, previous_updates,weight_decay=0):
        for l in range(1, my_network.n_layers):
            previous_updates["W" + str(l)] = beta * previous_updates["W" + str(l)] + (1 - beta) * np.square(
                my_network.grads["dW" + str(l)])
            previous_updates["b" + str(l)] = beta * previous_updates["b" + str(l)] + (1 - beta) * np.square(
                my_network.grads["db" + str(l)])
            factorW = eta / (np.sqrt(previous_updates["W" + str(l)] + epsilon))
            factorb = eta / (np.sqrt(previous_updates["b" + str(l)] + epsilon))
            my_network.theta["W" + str(l)] -= factorW * my_network.grads["dW" + str(l)] - eta*weight_decay*my_network.theta["W" + str(l)]
            my_network.theta["b" + str(l)] -= factorb * my_network.grads["db" + str(l)] -eta*weight_decay*my_network.theta["b" + str(l)]
            return previous_updates
            '''
            Working previously fetched an issue that the previous_updates should be returned
            if not then it is showing validation accuracy as 9.05%
            but after returning this slightly better
            '''

    @staticmethod
    def nadam(my_network,eta, beta1, beta2, epsilon, M, V, t,weight_decay=0):
        for l in range(1, my_network.n_layers):
            M["W" + str(l)] = beta1 * M["W" + str(l)] + (1 - beta1) * my_network.grads["dW" + str(l)]
            M["b" + str(l)] = beta1 * M["b" + str(l)] + (1 - beta1) * my_network.grads["db" + str(l)]
            MW_corrected = M["W" + str(l)] / (1 - (beta1 ** (t)))
            Mb_corrected = M["b" + str(l)] / (1 - (beta1 ** (t)))

            V["W" + str(l)] = beta2 * V["W" + str(l)] + (1 - beta2) * np.square(my_network.grads["dW" + str(l)])
            V["b" + str(l)] = beta2 * V["b" + str(l)] + (1 - beta2) * np.square(my_network.grads["db" + str(l)])
            VW_corrected = V["W" + str(l)] / (1 - (beta2 ** (t)))
            Vb_corrected = V["b" + str(l)] / (1 - (beta2 ** (t)))

            factorW = eta / (np.sqrt(VW_corrected) + epsilon)
            factorb = eta / (np.sqrt(Vb_corrected) + epsilon)
            term1 = 1 - (beta1 ** (t))
            term2 = (1 - beta1) * my_network.grads["dW" + str(l)] / term1
            term3 = (1 - beta1) * my_network.grads["db" + str(l)] / term1
            my_network.theta["W" + str(l)] -= factorW * (beta1 * MW_corrected + term2) -eta*weight_decay*my_network.theta["W" + str(l)]
            my_network.theta["b" + str(l)] -= factorb * (beta1 * Mb_corrected + term3) -eta*weight_decay*my_network.theta["b" + str(l)]
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

  '''
    Initilizing the followings parameters with default values, which can be changed by the user:

    mode_of_initialization: The method used to initialize the weights and biases of the neural network
    n_layers: The number of layers in the neural network
    activation_function: The activation function used in the hidden layers of the neural network
    n_input: The number of input neurons in the input layer of the neural network
    n_output: The number of output neurons in the output layer of the neural network
    n_neurons: The number of neurons in each hidden layer of the neural network
    TrainInput: The input data used to train the neural network
    TrainOutput: The output data used to train the neural network
    ValInput: The input data used to validate the neural network
    ValOutput: The output data used to validate the neural network
    theta: The parameters of the neural network
    cache: The cache of the neural network
    grads: The gradients of the neural network
  
  '''
  
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



  def __init__(self,mode_of_initialization="random",number_of_hidden_layers=1,num_neurons_in_hidden_layers=4,activation="sigmoid",TrainInput=x_train_T,TrainOutput=y_train_T,ValInput=x_val_T,ValOutput=y_val_T):
    self.mode_of_initialization = mode_of_initialization
    neuronsPerLayer = []
    for i in range(number_of_hidden_layers):
      neuronsPerLayer.append(num_neurons_in_hidden_layers)
    self.n_layers = number_of_hidden_layers + 2
    self.activation_function = activation
    self.TrainInput = TrainInput
    self.TrainOutput = TrainOutput
    self.n_input = TrainInput.shape[0]
    self.n_output = TrainOutput[0,TrainOutput.argmax(axis = 1)[0]] + 1
    self.n_neurons = neuronsPerLayer
    self.n_neurons.append(self.n_output)
    self.n_neurons.insert(0 , self.n_input)
    self.cache["H0"] = TrainInput
    self.cache["A0"] = TrainInput
    self.grads = {}
    self.ValInput = ValInput
    self.ValOutput = ValOutput
    for l in range(1,self.n_layers):
      if self.mode_of_initialization == "random":
        self.theta["W" + str(l)] = np.random.randn(self.n_neurons[l] , self.n_neurons[l - 1])
      elif self.mode_of_initialization == "Xavier":
        limit = np.sqrt(2 / float(self.n_neurons[l - 1] + self.n_neurons[l]))
        self.theta["W" + str(l)] = np.random.normal(0.0, limit, size=(self.n_neurons[l],self.n_neurons[l - 1]))
      self.theta["b" + str(l)] = np.zeros((self.n_neurons[l] , 1))


  def forward(self, X, activation, theta):
    '''
    The forward function is used to calculate the output of the neural network using the input data and the parameters of the neural network.

    parameters:

    X: The input data to the neural network
    activation: The activation function used in the hidden layers of the neural network
    theta: The parameters of the neural network

    returns:
    y_hat: The output of the neural network
    '''
    self.cache["H0"] = X
    for l in range(1, self.n_layers):
        H = self.cache["H" + str(l - 1)]
        W = self.theta["W" + str(l)]
        b = self.theta["b" + str(l)]
        A = np.dot(W, H) + b
        self.cache["A" + str(l)] = A
        H = Util.apply_activation(A, activation)
        self.cache["H" + str(l)] = H
    Al = self.cache["A" + str(self.n_layers - 1)]
    y_hat= Compute.softmax(Al)

    return y_hat


  def backpropagation(self, y_predicted, e_y, batch_size, loss, activation, theta):
        '''
        This function is used to calculate the gradients of the weights and biases of the neural network using the backpropagation algorithm.

        paremeters:
        y_predicted: The output predicted by the neural network
        e_y: The true output of the neural network
        batch_size: The size of the batch for each iteration of training
        loss: The loss function to be minimized
        activation: The activation function used in the hidden layers of the neural network
        theta: The parameters of the neural network

        '''
        if loss == 'cross_entropy':
            dA = y_predicted - e_y
        elif loss=='mean_squared_error':
            dA=(y_predicted - e_y)*Compute.softmax_derivative(self.cache["A" + str(self.n_layers - 1)])
        m = dA.shape[1]
        self.grads["dA" + str(self.n_layers - 1)] = dA

        for k in range(self.n_layers - 1, 0, -1):
            dA = self.grads["dA" + str(k)]
            H_prev = self.cache["H" + str(k - 1)]
            A_prev = self.cache["A" + str(k - 1)]
            W = self.theta["W" + str(k)]

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
    for l in range(1 , self.n_layers):
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
        yPredicted = self.forward(self.TrainInput[:,i:i + batch_size],self.activation_function,theta)
        e_y = np.transpose(np.eye(self.n_output)[self.TrainOutput[0,i : i + batch_size]])
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
      train_acc = Util.accuracy(self.TrainInput, self.TrainOutput,y_hat)
      tarin_acc_per_epoch.append(train_acc)

      val_acc = Util.accuracy(self.ValInput, self.ValOutput,valy_hat)
      val_acc_per_epoch.append(val_acc)
    #   print(np.eye(self.n_output)[self.ValOutput[0]].T.shape,valy_hat.shape)

      print("---------"*20)
      print(f"Epoch  = {format(count+1)}")
      print(f"Training Accuracy = {format(tarin_acc_per_epoch[-1])}")
      print(f"Validation Accuracy = {format(val_acc_per_epoch[-1])}")
      wandb.log({"training_accuracy": train_acc,"validation_accuracy": val_acc,"training_loss":train_cost,"validation_loss": val_cost,"epoch": count})
    return train_c_epoch,tarin_acc_per_epoch,val_c_per_epoch,val_acc_per_epoch



my_network = MyNeuralNetwork(mode_of_initialization="Xavier",number_of_hidden_layers=3,num_neurons_in_hidden_layers=128,activation="sigmoid",TrainInput=x_train_T,TrainOutput=y_train_T,ValInput=x_val_T,ValOutput=y_val_T)
train=my_network.compute(eta = 0.0001,mom=0.5,beta = 0.9,beta1 = 0.9,beta2 = 0.9,epsilon =1e-9, optimizer = 'rmsprop',batch_size = 32,weight_decay=0.5,loss = 'mean_squared_error',epochs = 5)


