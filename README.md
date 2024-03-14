

## Description

This codebase presents a neural network class designed for classification tasks. It offers functionalities for initializing weights, forward propagation, and backpropagation. Various optimization algorithms such as stochastic gradient descent (SGD), momentum, Nesterov accelerated gradient (NAG), RMSprop, Adam, and Nadam are implemented to update the network parameters. The training script `train.py` allows for easy training of a classification model using either the `mnist` or `fashion_mnist` dataset. Additionally, the script facilitates customization of parameters through user input or via default values.

## Training Model

Execute `train.py` to train a classification model, specifying the dataset (`mnist` or `fashion_mnist`) and customizing parameters as needed. After training, it generates five plots illustrating training accuracy, validation accuracy, training cost, validation cost, and epochs with step count.

## Question Tasks

Each question script (e.g., `Question1.py`, `Question2.py`, etc.) corresponds to specific tasks using the `fashion_mnist` dataset, providing insights into various aspects of the neural network model and its performance.

## Script Descriptions

- `Question1.py`: Displays images of one object from each of the 10 classes using `sweep.log`.
- `Question2.py`: Implements the forward function, initializing parameters, and predicting class probabilities for input samples.
- `Question3.py`: Implements the backpropagation algorithm to compute parameter gradients for each layer. Offers various optimization algorithms for gradient descent.
- `Question4.py`, `Question5`, and `Question6`: Utilizes sweep for hyperparameter tuning to achieve optimal model configurations. Generates multiple plots for analysis and Observations are drawn from the graphs obtained in previous stages.
- `Question7.py`: Creates a confusion matrix using sweep and provides observations.
- `Question8.py`: Incorporates mean squared error loss alongside categorical cross-entropy loss. Uses optimal model configurations obtained from sweep for analysis.
- `Question10.py`: Demonstrates achieving good accuracy on the `mnist` dataset using only three hyperparameter configurations.



## Deployment

To deploy this project run `train.py` using,


```
python train.py --wandb_entity myname --wandb_project myprojectname
```

but before running the project you need need to run the followings to satisfy your module needs,

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

For google-colab or kaggle or any other notebooks you can use the following commands

```
!pip install numpy
!pip install wandb
!pip install sklearn
!pip install matplotlib
!pip install warnings
!pip install keras
!pip install tensorflow

```

**Note:** We are not using `tensorflow` in our code but it is required for wandb

the above code will work with the default value that s=is given into the default value parameters. And in the place of `myname`, please select a wandb entity whether it is **team name** or **single user** name or **org** name.  you can find this in the following link https://wandb.ai/settings

Next, you can customize parameters according to your choice like for an example let say you want to do 5 `epochs` so we need to use the following code,

```
python train.py --wandb_entity myname --wandb_project myprojectname --epochs 5`
```
or,

```
python train.py --wandb_entity myname --wandb_project myprojectname -e 5
```

you can use multiple parameters to get your desired output.Let say you want to do 5 `epochs` having 4 layers of neurons and in each layer there are 64 nodes with the activation function as tanh, then you need to use the following code,
```
python train.py --wandb_entity myname --wandb_project myprojectname --num_layers 4 --hidden_size 64 --activation "tanh"
```

or, 

```
python train.py --wandb_entity myname --wandb_project myprojectname -nhl 4 -sz 64 -a "tanh"
```

Given the following where in Name column two commands are given for each hyperparameters and also for dataset. you need write the commands followed by the arguments

| Name                      | Default Value | Description                                                              |
|---------------------------|---------------|--------------------------------------------------------------------------|
| `--wandb_entity` (`-we`)  | myname        | Wandb Entity used to track experiments in the Weights & Biases dashboard.|
| `--wandb_project` (`-wp`) | myprojectname | Project name used to track experiments in Weights & Biases dashboard.    |
| `--dataset` (`-d`)        | fashion_mnist | Dataset to be used for training. Choices: ["mnist", "fashion_mnist"].   |
| `--epochs` (`-e`)         | 1             | Number of epochs to train the neural network.                           |
| `--batch_size` (`-b`)     | 4             | Batch size used to train the neural network.                            |
| `--loss` (`-l`)           | cross_entropy | Loss function to be used. Choices: ["mean_squared_error", "cross_entropy"]. |
| `--optimizer` (`-o`)      | sgd           | Optimization algorithm to be used. Choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]. |
| `--learning_rate` (`-lr`) | 0.1           | Learning rate used to optimize model parameters.                       |
| `--momentum` (`-m`)       | 0.5           | Momentum used by momentum and nag optimizers.                           |
| `--beta`                   | 0.5           | Beta used by rmsprop optimizer.                                         |
| `--beta1`, `--beta2`       | 0.5           | Beta1 and Beta2 used by adam and nadam optimizers.                      |
| `--epsilon` (`-eps`)      | 0.000001      | Epsilon used by optimizers.                                             |
| `--weight_decay` (`-w_d`) | .0            | Weight decay used by optimizers.                                        |
| `--weight_init` (`-w_i`)  | random        | Weight initialization method. Choices: ["random", "Xavier"].            |
| `--num_layers` (`-nhl`)   | 1             | Number of hidden layers used in feedforward neural network.            |
| `--hidden_size` (`-sz`)   | 4             | Number of hidden neurons in a feedforward layer.                       |
| `--activation` (`-a`)     | sigmoid       | Activation function used. Choices: ["identity", "sigmoid", "tanh", "ReLU"] |


## Report

The report containing detailed observations and insights can be accessed [here](https://wandb.ai/abhirupmjdr_dl/deep-learning/reports/CS6910-Assignment-1--Vmlldzo2OTEyMjk3)

## Acknowledgements

 - [CS6910: Deep Learning](https://www.cse.iitm.ac.in/~miteshk/CS6910.html)
 - [README templetes](https://github.com/matiassingers/awesome-readme)
 - Instructed by [Mitesh Khapra](https://www.cse.iitm.ac.in/~miteshk/)
 - Deep Learning TAs

## FAQ

#### Is this code platfrom independent?

Yes, you can run it in your Mac, Windows, or linux machine. Also you can use it in google-colab or kaggle

#### Do we need to download all `.py` files?

No, only `train.py` file is sufficient but you can also see the implementation from other files.

#### Can we add more functions ?

Yes, you can as many optimization and/or activation functions you want to. You just need to write the implementationand pass the parameters.


