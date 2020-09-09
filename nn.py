import helper
import numpy as np
import matplotlib.pyplot as plt

'''
w = weights, b = bias, i = input, h = hidden, o = output, l = label
e.g. w_i_h = weights from input layer to hidden layer
'''
data, labels = helper.get_mnist()
w_i_h = np.random.uniform(-1, 1, (20, 784)) / 784
w_h_o = np.random.uniform(-1, 1, (10, 20)) / 20
b_i_h = np.random.uniform(-1, 1, (20, 1))
b_h_o = np.random.uniform(-1, 1, (10, 1))

learn_rate = 0.01
epochs = 2
batch_size = 1
acc = 0
for epoch in range(epochs):
    for i, l in zip(data, labels):
        i.shape += (1,)
        l.shape += (1,)
        # Forward propagation input -> hidden
        h_pre = b_i_h + w_i_h @ i
        h = 1 / (1 + np.exp(-h_pre))
        # Forward propagation hidden -> output
        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))

        # Error calculation
        e = np.sum((o - l) ** 2, axis=0)
        is_correct = np.argmax(o) == np.argmax(l)

        # Backpropagation output -> hidden (cost function derivative)
        delta_o = o - l
        w_h_o += -learn_rate * delta_o @ np.transpose(h)
        b_h_o += -learn_rate * np.sum(delta_o)
        # Backpropagation hidden -> input (activation function derivative)
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
        w_i_h += -learn_rate * delta_h @ np.transpose(i)
        b_i_h += -learn_rate * np.sum(delta_h)

        # Show accuracy for this epoch
        acc += is_correct
    print(f'Acc: {round((acc / data.shape[0]) * 100, 2)}%')
    acc = 0

# Show results
while True:
    image_number = int(input("Enter a number (0 - 59999): "))
    image_index = image_number
    plt.imshow(data[image_index].reshape(28, 28), cmap='Greys')

    # Forward propagation input -> hidden
    h_pre = b_i_h + w_i_h @ data[image_index].reshape(784, 1)
    h = 1 / (1 + np.exp(-h_pre))
    # Forward propagation hidden -> output
    o_pre = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(-o_pre))

    plt.title(f'Subscribe if its a {o.argmax()} :)')
    plt.show()
