import numpy as np


class Softmax:

    def __init__(self):
        self.output = np.array([[]])

    def forward(self, inputs):
        # Get not normalised probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalise them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities
