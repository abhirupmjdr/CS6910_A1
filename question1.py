#question 1
# !pip install wandb
# import wandb
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import numpy as np

# wandb.login()
# wandb.init(
#       # Set the project where this run will be logged
#       project="deep-learning",
#       # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
#       # Track hyperparameters and run metadata
#       config={
#       "epochs": 1
#       })

(train_input, train_output), (test_input,test_output) = fashion_mnist.load_data()

class_names = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

# Plot one sample image for each class
class_counts = np.bincount(train_output)
total=len(train_output)
for i in range(len(class_names)):
    # Find the first image with the corresponding class label
    idx=np.where(train_output==i)[0][0]
    # plotting the image
    plt.subplot(2,5,i+1)
    plt.imshow(train_input[idx],cmap='gray')
    plt.title(class_names[i])
    plt.axis("off")
'''
here wandb is used for logging the sample images
'''
wandb.log({"Sample Images": plt})

