import numpy as np

import first_neurons.activation_functions as activation_functions
from first_neurons.Layer_Dense import LayerDense

inputs = [[1, 2, 3, 2.5],
          [2., 5., -1., 2],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]


# We essentially have:
# 4 neurons on input layer
# 2 hidden layers with 3 neurons
# 1st hidden layer have 4 inputs, 3 neurons, shape (3, 4) - so it matches input layer with (3, 4) shape as well
# 2nd hidden layer have 3 inputs, 3 neurons, shape (3, 3)
def calc_two_layers_batched_samples():
    layer1_outputs = np.dot(np.array(inputs), np.array(weights).T) + biases
    layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

    return layer2_outputs


def calc_one_layer_entity_batched_samples(x_data):
    layer_dense1 = LayerDense(2, 3)
    layer_dense2 = LayerDense(3, 3)

    relu = activation_functions.ReLU()
    softmax = activation_functions.Softmax()

    # First pass through our first layer (input layer not included, as it does not transform the data)
    layer_dense1.forward(x_data)

    # Pass through ReLU to (hopefully) fit some nonlinear function
    relu.forward(layer_dense1.output)

    # Pass through second layer (output layer in this case)
    layer_dense2.forward(relu.output)

    # Make predictions with normalised exponential values from previous layer
    # Do we need to make sure that values passed here are not bigger than 1000 to avoid overflow and dead neurons?
    softmax.forward(layer_dense2.output)

    return softmax.output
