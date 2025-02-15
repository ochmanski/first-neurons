import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

import first_neurons.many_layers as many_layers
import first_neurons.single_layer as single_layer

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

plt.style.use('dark_background')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')

if __name__ == '__main__':
    print(single_layer.calc_layer_batched_samples())
    print(many_layers.calc_two_layers_batched_samples())
    print(many_layers.calc_two_layers_entity_batched_samples(X)[:5])

    # plt.show()
