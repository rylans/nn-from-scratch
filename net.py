'''Neural Net'''
from neuron import Neuron

class Net(object):
    '''Neural network composed of layers of Neurons

    >>> len(Net([5,7,11], 1).layers[0])
    5

    >>> len(Net([5,7,11], 1).layers[1])
    7

    >>> len(Net([5,7,11], 1).layers[2])
    11
    '''
    def __init__(self, schema, input_size):
        import math
        sigm = lambda x: (1.0/(1+pow(math.e, x)))
        sigmp = lambda x: sigm(x)*(1.0-sigm(x))
        error = lambda x, y: 0.5*pow(x-y, 2)

        self.layers = []
        self.input_size = input_size

        last_size = input_size
        for layer in schema:
            layer_neurons = [Neuron(last_size, sigm, sigmp, error) \
                    for k in range(layer)]
            last_size = layer
            self.layers.append(layer_neurons)

    def feedforward(self, inputs):
        '''Feed input through the network and return final activations

        >>> len(Net([2,4,7], 3).feedforward([0.1,0.2,0.3]))
        7
        '''
        layer_inputs = inputs
        for layer in self.layers:
            layer_activations = [n.out(layer_inputs) for n in layer]
            layer_inputs = layer_activations
        return layer_activations

    def __repr__(self):
        return "Net({0}, input dim={1})".\
                format(str([len(l) for l in self.layers]), self.input_size)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
