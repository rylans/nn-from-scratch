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
        all_layer_activations = []
        for layer in self.layers:
            layer_activations = [n.out(layer_inputs) for n in layer]
            layer_inputs = layer_activations
            all_layer_activations.append(layer_activations)
        return all_layer_activations

    def train_1(self, inputs, target):
        predictions = self.feedforward(inputs)

        previous_activations = predictions[:-1]
        previous_activations.reverse()

        layers_reversed = self.layers[::-1]
        #print "zip shape {0}".format([(len(k), len(j)) for k,j in zip(layers_reversed, previous_activations)])

        #Product of the previous error and previous output derivative
        error_der = [1.0 for k in previous_activations[0]]
        for layer, prev_activation in zip(layers_reversed, previous_activations):
            #activs = [p*e for p, e in zip(prev_activation, error_der)]
            target = target*error_der[0]
            error_der = []
            for neuron in layer:
                #print "prev activation len {0} for neuron: {1}".format(len(prev_activation), neuron)
                dE, dy = neuron.learn_1(prev_activation, target)
                error_der.append(dE*dy)

    def __repr__(self): return "Net({0}, input dim={1})".\
                format(str([len(l) for l in self.layers]), self.input_size)

def training_trial_1(steps):
    import random
    input_dim = 2
    net = Net([4, 2, 1], input_dim)
    print net.feedforward([0.1,0.8])[-1]
    for i in range(steps):
        ins = [random.random() for k in range(input_dim)]
        target = sum(ins)
        net.train_1(ins, target)
    print net.feedforward([0.1,0.8])[-1]

def training_trial_2(steps):
    import random
    input_dim = 2
    net = Net([4, 2, 1], input_dim)
    print net.feedforward([0.1,0.8])[-1]
    for i in range(steps):
        ins = [random.random() for k in range(input_dim)]
        target = min(ins)
        net.train_1(ins, target)
    print net.feedforward([0.1,0.8])[-1]

if __name__ == '__main__':
    import doctest
    doctest.testmod()

    training_trial_1(200)
    print
    training_trial_2(700)
