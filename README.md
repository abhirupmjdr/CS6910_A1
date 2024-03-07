# Assignment 1 (CS6910)

This codebase presents a neural network class designed for classification tasks. It offers functionalities for initializing weights, forward propagation, and backpropagation. Various optimization algorithms such as stochastic gradient descent (SGD), momentum, Nesterov accelerated gradient (NAG), RMSprop, Adam, and Nadam are implemented to update the network parameters. Here are the instructions for navigating through the code:

1. **Training Model:**
   - Execute `train.py` to train a classification model, choosing either the `mnist` or `fashion_mnist` dataset.
   - The script allows customization of parameters either through user input or via default values.
   - After training, it generates five plots illustrating training accuracy, validation accuracy, training cost, validation cost, and epochs with step count.

2. **Question Tasks:**
   - Each question (`Question1.py`, `Question2.py`, `Question3.py`, `Question4.py`, `Question7.py`, `Question8.py`, `Question10.py`) corresponds to specific tasks using the `fashion_mnist` dataset.
   - Running these scripts provides insights into various aspects of the neural network model and its performance.

3. **Script Descriptions:**
   - `Question1.py`: Displays images of one object from each of the 10 classes using `sweep.log`.
   - `Question2.py`: Implements the forward function, initializing parameters and predicting class probabilities for input samples.
   - `Question3.py`: Implements the backpropagation algorithm to compute parameter gradients for each layer. Offers various optimization algorithms for gradient descent.
   - `Question4.py`: Utilizes `sweep` for hyperparameter tuning to achieve optimal model configurations. Generates multiple plots for analysis.
   - `Question5` and `Question6`: Observations are drawn from the graphs obtained in previous stages.
   - `Question7.py`: Creates a confusion matrix using `sweep` and provides observations.
   - `Question8.py`: Incorporates mean squared error loss alongside categorical cross-entropy loss. Uses optimal model configurations obtained from sweep for analysis.
   - `Question10.py`: Demonstrates achieving good accuracy on the `mnist` dataset using only three hyperparameter configurations.

4. **Report Link:**
   - The report containing detailed observations and insights can be accessed via the provided link.

This codebase provides a comprehensive understanding of building and optimizing neural network models for classification tasks, along with thorough analysis and interpretation of results.
