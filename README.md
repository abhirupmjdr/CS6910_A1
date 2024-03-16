

## Description

This codebase presents a neural network class designed for classification tasks. It offers functionalities for initializing weights, forward propagation, and backpropagation. Various optimization algorithms such as stochastic gradient descent (SGD), momentum, Nesterov accelerated gradient (NAG), RMSprop, Adam, and Nadam are implemented to update the network parameters. The training script [`train.py`](https://github.com/abhirupmjdr/CS6910_A1/blob/main/train.py) allows for easy training of a classification model using either the `mnist` or `fashion_mnist` dataset. Additionally, the script facilitates customization of parameters through user input or via default values.

## Question Tasks

Each question script (e.g., [`Question1.py`](https://github.com/abhirupmjdr/CS6910_A1/blob/main/question1.py), [`Question2.py`](https://github.com/abhirupmjdr/CS6910_A1/blob/main/Question2.py), etc.) corresponds to specific tasks using the `fashion_mnist` dataset, providing insights into various aspects of the neural network model and its performance.
Given the question with description of the question,

- [`Question1.py`](https://github.com/abhirupmjdr/CS6910_A1/blob/main/question1.py): Displays images of one object from each of the 10 classes using `sweep.log()`.
- [`Question2.py`](https://github.com/abhirupmjdr/CS6910_A1/blob/main/Question2.py): Implements the forward function, initializing parameters, and predicting class probabilities for input samples.
- [`Question3.py`](https://github.com/abhirupmjdr/CS6910_A1/blob/main/Question3.py): Implements the backpropagation algorithm to compute parameter gradients for each layer. Offers various optimization algorithms for gradient descent.
- [`Question4.py`](https://github.com/abhirupmjdr/CS6910_A1/blob/main/Question4_5.py), [`Question5`](https://github.com/abhirupmjdr/CS6910_A1/blob/main/Question4_5.py), and [`Question6`](https://github.com/abhirupmjdr/CS6910_A1/blob/main/Question6.py): Utilizes sweep for hyperparameter tuning to achieve optimal model configurations. Generates multiple plots for analysis and Observations are drawn from the graphs obtained in previous stages.
- [`Question7.py`](https://github.com/abhirupmjdr/CS6910_A1/blob/main/Question7.py): Creates a confusion matrix using sweep and provides observations.
- [`Question8.py`](https://github.com/abhirupmjdr/CS6910_A1/blob/main/Question8.py): Incorporates mean squared error loss alongside categorical cross-entropy loss. Uses optimal model configurations obtained from sweep for analysis.
- [`Question10.py`](https://github.com/abhirupmjdr/CS6910_A1/blob/main/Question10.py): Demonstrates achieving good accuracy on the `mnist` dataset using only three hyperparameter configurations.

- [`train.py`](https://github.com/abhirupmjdr/CS6910_A1/blob/main/train.py): a code for training the dataset where just downloading this file we can change hyperparameres accordingly and it will the logged to the respective wandb account and project.

## Installation


Before running the project you need need to run the followings to satisfy your module needs,

For conda enviroment you can run the following,

```
conda install numpy
conda install wandb
conda install sklearn
conda install matplotlib
conda install warnings
conda install keras
conda install tensorflow
```

For google-colab or kaggle or any other notebooks you can use the following commands,

```
!pip install numpy
!pip install wandb
!pip install sklearn
!pip install matplotlib
!pip install warnings
!pip install keras
!pip install tensorflow

```

**Note:** We are not using `tensorflow` in our code but it is required for wandb.

## Running the model

To deploy this project run [`train.py`](https://github.com/abhirupmjdr/CS6910_A1/blob/main/train.py) using,


```
python train.py --wandb_entity myname --wandb_project myprojectname
```

The above code will work with the default value that is given into the default value parameters. The default parameters are given in such a way that it gives almost maximum accuracy among the combinations of hyperparameters. And in the place of `myname`, please select a wandb entity whether it is **team name** or **single user** name or **org** name.  you can find this in the following link [here](https://wandb.ai/settings). And in the project name field fill the project name where it will be logged

Next, you can customize parameters according to your choice like for example let say you want to do 5 `epochs` so we need to use the following code,

```
python train.py --wandb_entity myname --wandb_project myprojectname --epochs 5
```
or,

```
python train.py --wandb_entity myname --wandb_project myprojectname -e 5
```

you can use multiple parameters to get your desired output. Let say you want to do 5 `epochs` having 4 layers of neurons and in each layer there are 64 nodes with the activation function as tanh, then you need to use the following code,
```
python train.py --wandb_entity myname --wandb_project myprojectname --num_layers 4 --hidden_size 64 --activation "tanh"
```

or, 

```
python train.py --wandb_entity myname --wandb_project myprojectname -nhl 4 -sz 64 -a "tanh"
```

Here I have given another example of how every required hyperparameter can be manipulated,

```
python train.py --wandb_entity abhirupmjdr_dl --wandb_project cs6910-a1 -a 'tanh' -o 'adam' -nhl 3 -sz 32 -w_i 'Xavier' --beta1 0.9 --beta2 0.99 -lr 0.001 -e 5 -b 32
```
In the above code --wandb_entity value  is `abhirupmjdr_dl` and project is my that project where it will be logged which is `cs6910-a1`.

**Note:** If google colab or kaggle or jupyter notebook is used then use an `!` before the line. For an example running our last example in google colab will be like,
```
!python train.py --wandb_entity abhirupmjdr_dl --wandb_project cs6910-a1 -a 'tanh' -o 'adam' -nhl 3 -sz 32 -w_i 'Xavier' --beta1 0.9 --beta2 0.99 -lr 0.001 -e 5 -b 32

```

Given the following where in the Name column two commands are given for each hyperparameter and also for the dataset. you need to write the any of the two commands followed by the arguments.

| Name                      | Default Value | Description                                                              |
|---------------------------|---------------|--------------------------------------------------------------------------|
| `--wandb_entity` (`-we`)  | myname        | Wandb Entity used to track experiments in the Weights & Biases dashboard.|
| `--wandb_project` (`-wp`) | myprojectname | Project name used to track experiments in Weights & Biases dashboard.    |
| `--dataset` (`-d`)        | fashion_mnist | Dataset to be used for training. Choices: ["mnist", "fashion_mnist"].   |
| `--epochs` (`-e`)         | 10             | Number of epochs to train the neural network.                           |
| `--batch_size` (`-b`)     | 32             | Batch size used to train the neural network.                            |
| `--loss` (`-l`)           | cross_entropy | Loss function to be used. Choices: ["mean_squared_error", "cross_entropy"]. |
| `--optimizer` (`-o`)      | nadam           | Optimization algorithm to be used. Choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]. |
| `--learning_rate` (`-lr`) | 0.0001           | Learning rate used to optimize model parameters.                       |
| `--momentum` (`-m`)       | 0.5           | Momentum used by momentum and nag optimizers.                           |
| `--beta`                   | 0.9           | Beta used by rmsprop optimizer.                                         |
| `--beta1`, (`--beta2`)       | 0.9           | Beta1 and Beta2 used by adam and nadam optimizers.                      |
| `--epsilon` (`-eps`)      | 0.000001      | Epsilon used by optimizers.                                             |
| `--weight_decay` (`-w_d`) | .0            | Weight decay used by optimizers.                                        |
| `--weight_init` (`-w_i`)  | Xavier        | Weight initialization method. Choices: ["random", "Xavier"].            |
| `--num_layers` (`-nhl`)   | 4             | Number of hidden layers used in feedforward neural network.            |
| `--hidden_size` (`-sz`)   | 128             | Number of hidden neurons in a feedforward layer.                       |
| `--activation` (`-a`)     | tanh       | Activation function used. Choices: ["identity", "sigmoid", "tanh", "ReLU"] |


## Project Report (on Wandb)

The report containing detailed observations and insights can be accessed [here](https://wandb.ai/abhirupmjdr_dl/deep-learning/reports/CS6910-Assignment-1--Vmlldzo2OTEyMjk3).

## Acknowledgements

 - [CS6910: Deep Learning](https://www.cse.iitm.ac.in/~miteshk/CS6910.html)
 - [README templetes](https://github.com/matiassingers/awesome-readme)
 - Instructed by [Mitesh Khapra](https://www.cse.iitm.ac.in/~miteshk/)
 - Deep Learning TAs

## FAQ

#### 1. Is this code platfrom independent?

Yes, you can run it in your Mac, Windows, or linux machine. Also you can use it in google-colab or kaggle.

#### 2. Do we need to download all `.py` files?

No, only [`train.py`](https://github.com/abhirupmjdr/CS6910_A1/blob/main/train.py) file is sufficient but you can also see the implementation from other files.

#### 3. Can we add more functions ?

Yes, you can as many optimization and/or activation functions you want to. You just need to write the implementationand pass the parameters.

#### 4. What is copy.js?

It is not related to the Deep Neural Netowrk model, it is a script file which allows us to add a copy a button in the markdown (i.e. readme.md) file to copy with a single click. Although it is supported by most of the browsers but still for safety reason to work almost with all browsers I have added it.
