'''Neuron'''
class Neuron(object):
    LEARNING_RATE = 0.5

    def __init__(self, num_inputs, activation, activationp, error):
        self.weights = [self.rnd() for i in range(num_inputs + 1)]
        self.num_inputs = num_inputs
        self.activation = activation
        self.activationp  = activationp
        self.error = error
        self.last_error = 999

    def rnd(self):
        import random
        return (3.0 + (random.random() - 0.5))/6

    def out(self, inputs):
        if len(inputs) != self.num_inputs:
            raise ValueError('inputs has length {0} instead of {1}'.\
                    format(str(len(inputs)), str(self.num_inputs)))
        inputs_with_bias = [x for x in inputs] + [1]
        return self.activation(sum([w*x for w, x in zip(self.weights, inputs_with_bias)]))

    def learn_1(self, inputs, target):
        output = self.out(inputs)
        reg = self.regularize()
        this_error = target - output
        error_value = self.error(target, output)
        self.last_error = error_value
        delta_ws = [this_error*self.LEARNING_RATE*error_value*x for x in inputs] + [this_error*self.LEARNING_RATE*error_value]
        oldw = self.weights
        self.weights = [w - dw for w, dw in zip(self.weights, delta_ws)]

    def regularize(self):
        return 0.5*sum([pow(w, 2) for w in self.weights])

    def __repr__(self):
        return 'Neuron({0})'.format(self.weights)

def learn_specific(steps):
    import math
    import random
    sigm = lambda x: (1.0/(1+pow(math.e, x)))
    sigmp = lambda x: sigm(x)*(1.0-sigm(x))
    error = lambda x, y:  0.5*pow(x-y, 2)

    neuron = Neuron(3, sigm, sigmp, error)
    print neuron
    for i in range(steps):
        inputs = [random.random() for r in range(3)]
        target = 2*inputs[0] + 0.3*inputs[1] - 0.7*inputs[2]
        neuron.learn_1(inputs, target)
    print neuron
    print neuron.last_error

if __name__ == '__main__':
    learn_specific(200)
