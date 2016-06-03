'''perceptron'''
from random import randrange
from random import random

class Perceptron(object):
    def __init__(self, input_dimension, error_f):
        self._weights = [self.rnd() for i in range(input_dimension + 1)]
        self._input_dim = input_dimension
        self._error = error_f
        self._best_weights = self._weights
        self._best_error = None
        self.anneal = 0.99

    def rnd(self):
        import random
        return (1.0 + (random.random() - 0.5))/2

    def adjust(self, w):
        from random import choice
        self.anneal *= 0.99
        if self.anneal < 0.05:
            self.anneal = 0.05
        return w * choice([-1.41, 0.709, 1.41, -0.709]) * self.anneal

    def out(self, inputs):
        if len(inputs) != self._input_dim:
            raise ValueError('inputs has length {0} instead of {1}'.\
                    format(str(len(inputs)), str(self._input_dim)))
        inputs_with_bias = [x for x in inputs] + [1]
        return sum([w*x for w, x in zip(self._weights, inputs_with_bias)])

    def learn_1(self, inputs, target):
        prev_error = self.out_error(inputs, target)
        if self._best_error == None:
            self._best_error = prev_error

        #randomly modify a weight
        rand_ix = randrange(0, self._input_dim + 1)
        orig_w = self._weights[rand_ix]
        adj_w = self.adjust(self._weights[rand_ix])
        self._weights[rand_ix] = adj_w

        new_error = self.out_error(inputs, target)

        print prev_error ," ,  ", new_error

        if prev_error < new_error:
            self._weights[rand_ix] = orig_w

        return new_error

    def out_error(self, inputs, target):
        output = self.out(inputs)
        error_value = self._error(target, output)
        return error_value

def test_n1_identity():
    p = Perceptron(1, lambda x, y: (x-y)*(x-y))

    for i in range(120):
        if i%10 == 0:
            print p._weights , ' anneal=', p.anneal
        v = random() * 30
        p.learn_1([v], v)
        p.learn_1([v+0.66], v+0.66)

    print 'expect 12: ', p.out([12.0])

if __name__ == '__main__':
    #p = Perceptron(2, lambda x, y: (x-y)*(x-y))

    test_n1_identity()

    '''
    #xor problem
    for i in range(15):
        print p.learn_1([0.1, 0.1], (1- (0.1*0.1)))
        print p.learn_1([0.88, 0.92], (1- (0.88*0.92)))
    '''

    '''
    for i in range(30):
        r1 = random() * 50
        r2 = (random() - 0.5) * 35
        t = r1 + r2
        p.learn_1([r1, r2], t)

    print p._weights
    '''
