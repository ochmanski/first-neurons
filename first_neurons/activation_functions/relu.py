import numpy as np


class ReLU:
    def __init__(self):
        self.output = np.array([[]])

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
