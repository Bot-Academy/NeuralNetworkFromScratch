import numpy as np
import helper


class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate, activation_functions):
        self.learning_rate = learning_rate

        self.layers = []
        self.layers.append({
            'num_neurons': layer_sizes[0],
            'neuron_values': None,
        })
        for idx in range(1, len(layer_sizes)):
            self.layers.append({
                'num_neurons': layer_sizes[idx],
                'pre_neuron_values': np.empty(layer_sizes[idx], dtype=list),
                'neuron_values': np.empty(layer_sizes[idx], dtype=list),
                'weights': np.random.uniform(-1, 1, (layer_sizes[idx], layer_sizes[idx-1])) / layer_sizes[idx-1],
                'bias': np.random.uniform(-1, 1, (layer_sizes[idx], 1)),
                'activation': activation_functions[idx-1],
            })

    def train(self, data, labels, batch_size, epochs):
        acc = 0
        for epoch in range(epochs):
            print(f'Epoch: {epoch}')
            for i in range(0, len(data), batch_size):
                data_batch = np.transpose(data[i:i + batch_size])
                label_batch = np.transpose(labels[i:i + batch_size])
                self.layers = forward_propagation(self.layers, data_batch)
                error = calculate_error(self.layers[-1], label_batch, batch_size)
                # print(f'Error: {np.sum(error)}')
                acc += np.sum(np.argmax(self.layers[-1]['neuron_values'], axis=0) == np.argmax(label_batch, axis=0)) / batch_size
                self.layers = backpropagation(self.layers, label_batch, self.learning_rate)
            print(f'Acc: {round((acc / data.shape[0]) * 100, 2)}%')
            acc = 0


def forward_propagation(layers, inputs):
    for idx, layer in enumerate(layers):
        if idx == 0:
            layers[idx]['neuron_values'] = inputs
        else:
            layers[idx]['pre_neuron_values'] = layers[idx]['bias'] + \
                                               layers[idx]['weights'] @ layers[idx-1]['neuron_values']
            layers[idx]['neuron_values'] = apply_activation_function(layer)

    return layers


def apply_activation_function(layer):
    values = layer['pre_neuron_values']

    if layer['activation'] == 'sigmoid':
        return 1 / (1 + np.exp(-values))
    elif layer['activation'] == 'softmax':
        e_x = np.exp(values - np.transpose(np.array([np.max(values, axis=1)])))
        return e_x / np.transpose(np.array([e_x.sum(axis=1)]))
    else:
        print("Unknown activation function!")


def apply_activation_function_derivative(layer):
    activations = apply_activation_function(layer)

    if layer['activation'] == 'sigmoid':
        return activations * (1 - activations)
    elif layer['activation'] == 'softmax':
        activations_vector = activations.reshape(activations.shape[0], 1)
        activations_matrix = np.tile(activations_vector, activations.shape[0])
        return np.diag(activations) - (activations_matrix * np.transpose(activations_matrix))

    else:
        print("Unknown activation function!")


def calculate_error(last_layer, labels, batch_size):
    if last_layer['error_function'] == 'mean_squared_error':
        return (1 / batch_size) * np.sum((last_layer['neuron_values'] - labels) ** 2, axis=0)
    else:
        print("Unknown activation function!")


def backpropagation(layers, labels, learning_rate):
    last_delta = None
    for idx in range(len(layers) - 1, 0, -1):
        if idx == len(layers) - 1:
            delta = layers[idx]['neuron_values'] - labels
            jacobian_output_weights = delta @ np.transpose(layers[idx-1]['neuron_values'])
            jacobian_output_bias = np.sum(delta)
            layers[idx]['weights'] += -learning_rate * jacobian_output_weights
            layers[idx]['bias'] += -learning_rate * jacobian_output_bias
        else:
            delta = np.transpose(layers[idx+1]['weights']) @ last_delta * apply_activation_function_derivative(layers[idx])
            jacobian_hidden_weights = delta @ np.transpose(layers[idx-1]['neuron_values'])
            jacobian_hidden_bias = np.sum(delta)
            layers[idx]['weights'] += -learning_rate * jacobian_hidden_weights
            layers[idx]['bias'] += -learning_rate * jacobian_hidden_bias

        last_delta = delta
    return layers


nn = NeuralNetwork(layer_sizes=[784, 50, 10], learning_rate=0.01,
                   activation_functions=['sigmoid', 'sigmoid'])
data, labels = helper.get_mnist()
nn.train(data, labels, batch_size=1, epochs=3)
