'''Neuron'''
class Neuron(object):
    def __init__(self, num_inputs, activation, activationp):
        import random
        self.weights = [random.random() for i in range(num_inputs + 1)]
        self.num_inputs = num_inputs
        self.activation = activation
        self.activationp  = activationp

    def out(self, inputs):
        if len(inputs) != self.num_inputs:
            raise ValueError('inputs has length {0} instead of {1}'.\
                    format(str(len(inputs)), str(self.num_inputs)))
        inputs_with_bias = [x for x in inputs] + [1]
        return self.activation(sum([w*x for w, x in zip(self.weights, inputs_with_bias)]))

if __name__ == '__main__':
    import math
    sigm = lambda x: (1.0/(1+pow(math.e, x)))
    sigmp = lambda x: sigm(x)*(1.0-sigm(x))

    neuron = Neuron(5, sigm, sigmp)
    print neuron.out([0.1, 0.2, 0.3, 0.4, 0.22])
