
!pip install wandb

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

# sweep_config = {
#     'method': 'random'
# }
# metric = {
#     'name': 'val_acc_per_epoch',
#     'goal': 'maximize'
#     }

# Initialize wandb
wandb.login()

# Initialize wandb
# wandb.login()

# sweep_config = {
#     'method': 'random',  # Possible methods: 'grid', 'random', 'bayes'
#     'metric': {
#         'name': 'validation_accuracy',
#         'goal': 'maximize'
#     },
#     'parameters': {
#         'learning_rate': {
#             'min': 0.0001,
#             'max': 0.01
#         },
#         'batch_size': {
#             'values': [16, 32, 64, 128]
#         },
#         'optimizer':{
#             'values': ['adam','nadam','sgd','RMSprop','nag']
#         },
#         'activation':{
#             'values': ['sigmoid','tanh','relu']
#         },
#         'number_of_hidden_layers':{
#             'values':[3,4,5]
#         },
#         'num_neurons_in_hidden_layers':{
#             'values':[16,32,64,128]
#         },
#         'mode_of_initialization':{
#             'values':['random','xavier']
#         },
#         'epoch':{
#             'values':[5,10]
#         }
#     }
# }

# sweep_id = wandb.sweep(sweep_config, project="my-project")

# Define your model here


# sweep_config['metric'] = metric

# parameters_dict = {
#     'optimizer': {
#         'values': ['adam', 'nadam', 'sgd']
#         },
#     'number_of_hidden_layers':{
#         'values': [3,4,5]
#     },
#     'num_neurons_in_hidden_layers': {
#         'values': [16, 32, 64,128]
#         },
#     'activation': {
#           'values': ['sigmoid', 'tanh', 'relu']
#         },
#     'epoch':{
#         'values': [5,10,15]
#     }
# }

# sweep_config['parameters'] = parameters_dict

# parameters_dict.update({
#     'eta': {
#         # a flat distribution between 0 and 0.1
#         'distribution': 'uniform',
#         'min': 0,
#         'max': 0.1
#       },
#     'batch_size': {
#         # integers between 32 and 256
#         # with evenly-distributed logarithms
#         'distribution': 'q_log_uniform_values',
#         'q': 8,
#         'min': 32,
#         'max': 256,
#       }
#     })

# import pprint

# pprint.pprint(sweep_config)
# sweep_id = wandb.sweep(sweep_config, project="deep-learning")

from keras.datasets import fashion_mnist
(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()


'''normalizing the data'''
x_test = x_test / 255.0
x_train = x_train / 255.0


'''train set,val set ,test set split'''
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
            elif activation == 'relu':
                return Compute.Relu(A)
            elif activation == 'tanh':
                return Compute.tanh(A)

    @staticmethod
    def loss(input, true_output, predicted_output, loss, batch_size,n_output):
        if loss == 'cross_entropy':
            one_hot_true_output = np.eye(n_output)[true_output[0]].T
            return -np.sum(one_hot_true_output * np.log(predicted_output + 1e-9)) / batch_size


        if loss=='squared_loss':
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
            elif activation == 'relu':
                return dH_prev * Compute.Relu_derivative(A_prev)


class Update:
    @staticmethod
    def stochastic_gradient_descent(my_neetowrk,eta):
        for l in range(1, my_network.n_layers):
            W, dW = my_network.theta["W" + str(l)], my_network.grads["dW" + str(l)]
            b, db = my_network.theta["b" + str(l)], my_network.grads["db" + str(l)]
            W -= eta * dW
            b -= eta * db
            my_network.theta["W" + str(l)], my_network.theta["b" + str(l)] = W, b

    @staticmethod
    def nesterov_gradient_descent(my_network,i,eta, batch_size, beta, previous_updates,loss):
        theta = {}
        input_data = my_network.TrainInput[:, i:i + batch_size]
        output_data = my_network.TrainOutput[0, i:i + batch_size]
        for l in range(1, my_network.n_layers):
            theta["W" + str(l)] = my_network.theta["W" + str(l)] - beta * previous_updates["W" + str(l)]
            theta["b" + str(l)] = my_network.theta["b" + str(l)] - beta * previous_updates["b" + str(l)]
        y_predicted = my_network.forward(input_data, my_network.activation_function, my_network.theta)
        e_y = np.transpose(np.eye(my_network.n_output)[output_data])
        my_network.backpropagation(y_predicted, e_y, batch_size, loss, my_network.activation_function, my_network.theta)
        for l in range(1, my_network.n_layers):
            previous_updates["W" + str(l)] = beta * previous_updates["W" + str(l)] + (1-beta)*my_network.grads["dW" + str(l)]
            previous_updates["b" + str(l)] = beta * previous_updates["b" + str(l)] + (1-beta)*my_network.grads["db" + str(l)]
            my_network.theta["W" + str(l)] -= eta * my_network.grads["dW" + str(l)]
            my_network.theta["b" + str(l)] -= eta * my_network.grads["db" + str(l)]
        return previous_updates

    @staticmethod
    def momentum_gradient_descent(my_network,eta, beta, previous_updates):
        for l in range(1, my_network.n_layers):
            uW, ub = previous_updates["W" + str(l)], previous_updates["b" + str(l)]
            W, dW = my_network.theta["W" + str(l)], my_network.grads["dW" + str(l)]
            b, db = my_network.theta["b" + str(l)], my_network.grads["db" + str(l)]
            uW = beta * uW + (1-beta) * dW
            ub = beta * ub + (1-beta) * db
            W -= eta * uW
            b -= eta * ub
            previous_updates["W" + str(l)], previous_updates["b" + str(l)] = uW, ub
            my_network.theta["W" + str(l)], my_network.theta["b" + str(l)] = W, b
            return previous_updates

    @staticmethod
    def rms_prop(my_network,eta, beta, epsilon, previous_updates):
        for l in range(1, my_network.n_layers):
            previous_updates["W" + str(l)] = beta * previous_updates["W" + str(l)] + (1 - beta) * np.square(
                my_network.grads["dW" + str(l)])
            previous_updates["b" + str(l)] = beta * previous_updates["b" + str(l)] + (1 - beta) * np.square(
                my_network.grads["db" + str(l)])
            factorW = eta / (np.sqrt(previous_updates["W" + str(l)] + epsilon))
            factorb = eta / (np.sqrt(previous_updates["b" + str(l)] + epsilon))
            my_network.theta["W" + str(l)] -= factorW * my_network.grads["dW" + str(l)]
            my_network.theta["b" + str(l)] -= factorb * my_network.grads["db" + str(l)]
            return previous_updates
            '''
            Working previously fetched an issue that the previous_updates should be returned
            if not then it is showing validation accuracy as 9.05%
            but after returning this slightly better
            '''

    @staticmethod
    def nadam(my_network,eta, beta1, beta2, epsilon, M, V, t):
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
            my_network.theta["W" + str(l)] -= factorW * (beta1 * MW_corrected + term2)
            my_network.theta["b" + str(l)] -= factorb * (beta1 * Mb_corrected + term3)
        return M, V

    @staticmethod
    def adam(my_network,eta, beta1, beta2, epsilon, M, V, t): #taken from slide-2 page 42 [cs6910]
        for l in range(1, my_network.n_layers):
            M["W" + str(l)] = beta1 * M["W" + str(l)] + (1 - beta1) * my_network.grads["dW" + str(l)]
            M["b" + str(l)] = beta1 * M["b" + str(l)] + (1 - beta1) * my_network.grads["db" + str(l)]
            V["W" + str(l)] = beta2 * V["W" + str(l)] + (1 - beta2) * np.square(my_network.grads["dW" + str(l)])
            V["b" + str(l)] = beta2 * V["b" + str(l)] + (1 - beta2) * np.square(my_network.grads["db" + str(l)])
            MW_hat = M["W" + str(l)] / (1 - np.power(beta1,t))
            Mb_hat = M["b" + str(l)] / (1 - np.power(beta1,t))
            VW_hat = V["W" + str(l)] / (1 - np.power(beta2,t))
            Vb_hat = V["b" + str(l)] / (1 - np.power(beta2,t))
            my_network.theta["W" + str(l)] -= (eta / (np.sqrt(VW_hat) + epsilon)) * MW_hat
            my_network.theta["b" + str(l)] -= (eta / (np.sqrt(Vb_hat) + epsilon)) * Mb_hat
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
      elif self.mode_of_initialization == "xavier":
        limit = np.sqrt(2 / float(self.n_neurons[l - 1] + self.n_neurons[l]))
        self.theta["W" + str(l)] = np.random.normal(0.0, limit, size=(self.n_neurons[l],self.n_neurons[l - 1]))
      self.theta["b" + str(l)] = np.zeros((self.n_neurons[l] , 1))


  def forward(self, X, activation, theta):
    self.cache["H0"] = X
    for l in range(1, self.n_layers):
        H = self.cache["H" + str(l - 1)]
        W = self.theta["W" + str(l)]
        b = self.theta["b" + str(l)]
        A = np.dot(W, H) + b
        self.cache["A" + str(l)] = A
        H = Util.apply_activation(A, activation)
        self.cache["H" + str(l)] = H
        # print("----"*5,l,"------"*5)
        # print("A")
        # print(H)
        # print('H')
        # print(H)
    Al = self.cache["A" + str(self.n_layers - 1)]
    y_hat= Compute.softmax(Al)

    return y_hat


  def backpropagation(self, y_predicted, e_y, batch_size, loss, activation, theta):
        if loss == 'cross_entropy':
            dA = y_predicted - e_y
        elif loss=='squared_loss':
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
            # print("----"*5,k,"-----"*5)
            # print("dw")
            # print(dW)
            # print("db")
            # print(db)

        return




  def compute(self, eta = 0.1,beta = 0.5,beta1 = 0.5,beta2 = 0.5 ,epsilon = 0.000001, optimizer = 'sgd',batch_size = 4,loss = 'cross_entropy',epochs = 1):
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
            Update.stochastic_gradient_descent(self,eta) #working
        elif optimizer == 'nag':
            previous_updates=Update.nesterov_gradient_descent(self,i,eta, batch_size, beta, previous_updates,loss) #working

        elif optimizer == 'momentum': #referred from slide 43
          previous_updates=Update.momentum_gradient_descent(self,eta,beta,previous_updates) #working

        elif optimizer == 'RMSprop':
          previous_updates=Update.rms_prop(self,eta,beta,epsilon,previous_updates) #working
        elif optimizer == 'adam':
          epsilon = 1e-10
          M , V = Update.adam(self,eta,beta1,beta2,epsilon,M , V , count+1)
        elif optimizer == 'nadam':
          epsilon = 1e-8
          M , V = Update.nadam(self,eta,beta1,beta2,epsilon,M , V , count+1)

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
      conf_matrix = confusion_matrix(np.argmax(np.eye(self.n_output)[self.ValOutput[0]].T, axis=0),
                                np.argmax(valy_hat, axis=0))

      '''
        # Print confusion matrix
      cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["T-shirt/top",
                                                                        "Trouser",
                                                                        "Pullover",
                                                                        "Dress",
                                                                        "Coat",
                                                                        "Sandal",
                                                                        "Shirt",
                                                                        "Sneaker",
                                                                        "Bag",
                                                                        "Ankle boot"])

        # Customize appearance
      cm_display.plot(cmap='Oranges')
      plt.xlabel('Predicted Label', fontsize=14)
      plt.ylabel('True Label', fontsize=14)
      plt.title('Confusion Matrix', fontsize=16)
      plt.xticks(rotation=45, ha='right')
      plt.tight_layout()
      wandb.log({"Confusion Matrix at epoch"+str(count+1) : plt})
      '''


      print("---------"*20)
      print(f"Epoch Number = {format(count+1)}")
      print(f"Training Accuracy = {format(tarin_acc_per_epoch[-1])}")
      print(f"Validation Accuracy = {format(val_acc_per_epoch[-1])}")
    #   if(count==1):
    #     print(self.cache["A1"])
      wandb.log({"training_accuracy": train_acc,"validation_accuracy": val_acc,"training_loss":train_cost,"validation_loss": val_cost,"epoch": count})
    return train_c_epoch,tarin_acc_per_epoch,val_c_per_epoch,val_acc_per_epoch


# my_network = MyNeuralNetwork(mode_of_initialization="xavier",number_of_hidden_layers=3,num_neurons_in_hidden_layers=64,activation="relu",TrainInput=x_train_T,TrainOutput=y_train_T,ValInput=x_val_T,ValOutput=y_val_T)
# train=my_network.compute(eta = 0.01,beta = 0.9,beta1 = 0.1,beta2 = 0.1 ,epsilon = 0.05, optimizer = 'nag',batch_size = 32,loss = 'cross_entropy',epochs = 5)
'''
The below code is for creating the sweep feature
'''

def train():
    wandb.init(project = 'deep-learning')
    config = wandb.config
    my_network = MyNeuralNetwork(mode_of_initialization=config.mode_of_initialization,number_of_hidden_layers=config.number_of_hidden_layers,num_neurons_in_hidden_layers=config.num_neurons_in_hidden_layers,activation=config.activation,TrainInput=x_train_T,TrainOutput=y_train_T,ValInput=x_val_T,ValOutput=y_val_T)
    my_network.compute(eta =config.eta,beta = config.beta,beta1 = 0.5,beta2 = 0.5 ,epsilon = 0.05, optimizer = config.optimizer,batch_size =config.batch_size,loss = config.loss,epochs = config.epochs)


sweep_config = {
    'method': 'grid',
    'name': 'accuracy sweep',
    'metric': {
        'goal': 'maximize',
        'name': 'validation_accuracy'
        },
    'parameters': {
        'mode_of_initialization': {'values': ['xavier','random']},
        'number_of_hidden_layers' : {'values' : [3,4,5]},
        'num_neurons_in_hidden_layers' : {'values' : [32,64,128]},

        'eta': {'values':[0.001,0.0001]},
        'beta' : {'values' : [0.05,0.9,0.99,0.999]},
        'optimizer' : {'values' : ['adam','nadam','nag','momentum']},

        'batch_size': {'values': [16,32,64]},
        'epochs': {'values': [5,10]},
        'loss' : {'values' : ['cross_entropy']},
        'activation' : {'values' : ['sigmoid','relu','tanh']},
        'weight_decay' : {'values' : [0,0.0005,0.5]}
       }
    }


sweep_id = wandb.sweep(sweep_config, project="deep-learning")
wandb.agent(sweep_id , function = train , count = 20)

