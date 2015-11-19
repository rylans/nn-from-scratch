from neuron import Neuron

import math
import random

sigm = lambda x: (1.0/(1+pow(math.e, x)))
sigmp = lambda x: sigm(x)*(1.0-sigm(x))
error = lambda x, y: 0.5*pow(x-y, 2)

def first_last_errors(errors):
    sq = int(math.sqrt(len(errors)))
    first_errors = sum(errors[:sq])/sq
    last_errors = sum(errors[-sq:])/sq
    return first_errors, last_errors

def decreasing_verdict(f, l):
    v = "No"
    if f > l:
        v = "Yes"
    return v

def report(errors):
    f,l = first_last_errors(errors)
    v = decreasing_verdict(f, l)
    return "Early avg error: {0}, Late avg error: {1}, decreasing: {2}\n".format(f, l, v)

def test_1(steps):
    weights = 3
    print "Linear combination of weights {0}, {1} steps".format(weights, steps)
    neuron = Neuron(weights, sigm, sigmp, error)
    errors = []
    for i in range(steps):
        inputs = [random.random() for r in range(weights)]
        target = 2*inputs[0] + 0.3*inputs[1] - 0.7*inputs[2]
        neuron.learn_1(inputs, target)
        errors.append(neuron.last_error)
    print report(errors)

def test_2(steps):
    weights = 1
    print "Target itself {0}, {1} steps".format(weights, steps)
    neuron = Neuron(weights, sigm, sigmp, error)
    errors = []
    for i in range(steps):
        inputs = [random.random() for r in range(weights)]
        target = inputs[0]
        neuron.learn_1(inputs, target)
        errors.append(neuron.last_error)
    print report(errors)

def test_3(steps):
    weights = 1
    print "Target converse {0}, {1} steps".format(weights, steps)
    neuron = Neuron(weights, sigm, sigmp, error)
    errors = []
    for i in range(steps):
        inputs = [random.random() for r in range(weights)]
        target = 1.0 - inputs[0]
        neuron.learn_1(inputs, target)
        errors.append(neuron.last_error)
    print report(errors)

def test_4(steps):
    weights = 40
    print "Target max - min {0}, {1} steps".format(weights, steps)
    neuron = Neuron(weights, sigm, sigmp, error)
    errors = []
    for i in range(steps):
        inputs = [random.random() for r in range(weights)]
        imax = max(inputs)
        imin = min(inputs)
        target = imax - imin
        neuron.learn_1(inputs, target)
        errors.append(neuron.last_error)
    print report(errors)

def test_5(steps):
    weights = 40
    print "Target sqrt(avg) {0}, {1} steps".format(weights, steps)
    neuron = Neuron(weights, sigm, sigmp, error)
    errors = []
    for i in range(steps):
        inputs = [random.random() for r in range(weights)]
        avg = sum(inputs)/len(inputs)
        target = math.sqrt(avg)
        neuron.learn_1(inputs, target)
        errors.append(neuron.last_error)
    print report(errors)


if __name__ == '__main__':
    test_1(500)
    test_2(500)
    test_3(500)
    test_4(500)
    test_5(6524)
