import first_neurons.many_layers as many_layers
import first_neurons.single_layer as single_layer

if __name__ == '__main__':
    print(single_layer.calc_layer_batched_samples())
    print(many_layers.calc_two_layers_batched_samples())
