'''Neural Net'''
from neuron import Neuron

class Net(object):
    def __init__(self, schema, input_size):
        import math
        sigm = lambda x: (1.0/(1+pow(math.e, x)))
        sigmp = lambda x: sigm(x)*(1.0-sigm(x))
        error = lambda x, y: 0.5*pow(x-y, 2)

        self.layers = []
        self.input_size = input_size

        last_size = input_size
        for layer in schema:
            layer_neurons = [Neuron(last_size, sigm, sigmp, error) for k in range(layer)]
            last_size = layer
            self.layers.append(layer_neurons)

    def feedforward(self, inputs):
        layer_inputs = inputs
        for layer in self.layers:
            layer_activations = [n.out(layer_inputs) for n in layer]
            layer_inputs = layer_activations
        return layer_activations

    def __repr__(self):
        return "Net({0}, input dim={1})".\
                format(str([len(l) for l in self.layers]), self.input_size)

if __name__ == '__main__':
    net = Net([3, 5, 1], 2)
    print net
    print net.feedforward([0.11, 0.89])

    print

    net = Net([2, 4, 6, 8, 6, 4, 2], 5)
    print net
    print net.feedforward([0.1, 0.55, 0.2, 0.3, 0.35])

