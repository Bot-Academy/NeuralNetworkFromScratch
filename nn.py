import numpy as np


class NeuralNetwork:
    def __init__(self, structure, learning_rate, activation_function='sigmoid', error_function='mean-squared-error'):
        self.activation_function = activation_function
        self.error_function = error_function
        self.learning_rate = learning_rate

        self.layer = np.empty(len(structure), dtype=dict)
        for idx in range(len(structure) - 1):
            self.layer[idx] = {
                'weights': np.random.random((structure[idx], structure[idx + 1])),
                'bias': np.random.random((structure[idx + 1]))
            }

        self.pre_activated_neuron_values = np.empty(len(self.layer), dtype=list)
        self.activated_neuron_values = np.empty(len(self.layer), dtype=list)

    def forward_propagation(self, inputs):
        for idx, layer in enumerate(self.layer):
            if idx == 0:
                self.pre_activated_neuron_values[idx] = inputs
                self.activated_neuron_values[idx] = inputs
            else:
                self.pre_activated_neuron_values[idx] = self.layer[idx-1]['bias'] + \
                                                        self.activated_neuron_values[idx-1] @ \
                                                        self.layer[idx-1]['weights']
                self.activated_neuron_values[idx] = self.apply_activation_function(self.pre_activated_neuron_values[idx])

        return self.activated_neuron_values[-1]

    def apply_activation_function(self, activation_inputs):
        if self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-activation_inputs))
        else:
            print("Unknown activation function!")

    def apply_activation_function_derivative(self, derivative_inputs):
        if self.activation_function == 'sigmoid':
            activations = self.apply_activation_function(derivative_inputs)
            return activations * (1 - activations)
        else:
            print("Unknown activation function!")

    def calculate_error(self, output, label):
        if self.error_function == 'mean-squared-error':
            return np.sum((output - label) ** 2)
        else:
            print("Unknown activation function!")

    def backpropagation(self, error):
        delta_output = np.array([error])
        jacobian_output_weights = self.activated_neuron_values[-1] @ delta_output
        jacobian_output_bias = np.sum(delta_output)
        self.layer[-2]['weights'] += - self.learning_rate * jacobian_output_weights
        self.layer[-2]['bias'] += - self.learning_rate * jacobian_output_bias

        delta_hidden = np.transpose(
            self.layer[-2]['weights']) * delta_output @ self.apply_activation_function_derivative(
            self.pre_activated_neuron_values[-2])
        jacobian_hidden_weights = self.activated_neuron_values[0] * delta_hidden
        jacobian_hidden_bias = np.sum(delta_hidden)
        self.layer[-3]['weights'] += - self.learning_rate * jacobian_hidden_weights
        self.layer[-3]['bias'] += - self.learning_rate * jacobian_hidden_bias

    def train(self, data, labels, epochs):
        for epoch in range(epochs):
            print(f'Epoch: {epoch}')
            for sample in data:
                output = self.forward_propagation(sample)
                error = self.calculate_error(output, labels)
                print(f'Error: {error}')
                self.backpropagation(error)
