import random
import numpy as np
from nn import NeuralNetwork


def get_mnist():
    with np.load('mnist.npz') as f:
        images, labels = f['x_train'], f['y_train']
    # Normalize training data
    images = images.astype("float32") / 255
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    # Labels -> One Hot encoding
    print("Shape before one-hot encoding: ", labels.shape)
    labels = np.eye(10)[labels]
    print("Shape after one-hot encoding: ", labels.shape)
    return images, labels


random.seed(10)
# data = np.array([[0.3, 0.1, 0.6, 0.6, 0.35], [0.7, 0.9, 0.3, 0.65, 0.9]])
# labels = [0.5, 0.6]
data = get_mnist()
nn = NeuralNetwork(layer_sizes=[784, 200, 10], activation_functions=['sigmoid', 'sigmoid'], learning_rate=0.001)
nn.train(data=data[0], labels=data[1], batch_size=1, epochs=2)
