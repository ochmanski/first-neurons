import numpy as np

inputs = [1.0, 2.0, 3.0, 2.5]
batched_inputs = [[1.0, 2.0, 3.0, 2.5],
                  [2.0, 5.0, -1.0, 2.0],
                  [-1.5, 2.7, 3.3, -0.8]]


def np_calc_neuron():
    weights = [0.2, 0.8, -0.5, 1.0]
    bias = 2.0

    outputs = np.dot(weights, inputs) + bias

    return outputs


def simple_calc_neurons():
    weights = [[0.2, 0.8, -0.5, 1],
               [0.5, -0.91, 0.26, -0.5],
               [-0.26, -0.27, 0.17, 0.87]]
    biases = [2, 3, 0.5]

    layer_outputs = []

    for neuron_weights, neuron_bias in zip(weights, biases):
        neuron_output = 0

        for n_input, weight in zip(inputs, neuron_weights):
            neuron_output += n_input * weight

        neuron_output += neuron_bias
        layer_outputs.append(neuron_output)

    return layer_outputs


def np_calc_neurons():
    weights = [[0.2, 0.8, -0.5, 1],
               [0.5, -0.91, 0.26, -0.5],
               [-0.26, -0.27, 0.17, 0.87]]
    biases = [2, 3, 0.5]

    layer_outputs = np.dot(weights, inputs) + biases

    return layer_outputs


def np_calc_matrix_product():
    a = [1, 2, 3]
    b = [2, 3, 4]

    a = np.array([a])
    b = np.array([b]).T

    return np.dot(a, b)


def np_calc_neurons_batched_samples():
    weights = [[0.2, 0.8, -0.5, 1.0],
               [0.5, -0.91, 0.26, -0.5],
               [-0.26, -0.27, 0.17, 0.87]]
    biases = [2.0, 3.0, 0.5]

    layer_outputs = np.dot(batched_inputs, np.array(weights).T) + biases

    return layer_outputs


if __name__ == '__main__':
    print(np_calc_neuron())
    print(simple_calc_neurons())
    print(np_calc_neurons())
    print(np_calc_matrix_product())
    print(np_calc_neurons_batched_samples())
