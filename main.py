import random
import numpy as np
from nn import NeuralNetwork

random.seed(10)
data = np.array([[0.3, 0.1, 0.6, 0.6, 0.35], [0.7, 0.9, 0.3, 0.65, 0.9]])
labels = [0.5, 0.6]
nn = NeuralNetwork(structure=[5, 5, 1], learning_rate=0.1)
nn.train(data=data, labels=labels, epochs=50)
