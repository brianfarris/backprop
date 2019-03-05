import numpy as np


class Vertex:
    def __init__(self, edges=[], name=None):
        self.name = name
        self.edges = edges
        self.value = None
        self.grad_value = 0


class Dot(Vertex):
    def func(self, inputs):
        x, y = inputs
        return np.dot(x, y)

    def grads(self, inputs):
        x, y = inputs
        return [y.T, x.T]

class Multiplication(Vertex):
    def func(self, inputs):
        x, y = inputs
        return x * y

    def grads(self, inputs):
        x, y = inputs
        return [y, x]


class Addition(Vertex):
    def func(self, inputs):
        x, y = inputs
        return x + y

    def grads(self, inputs):
        x, y = inputs
        return [np.ones(x.shape), np.ones(y.shape)]


class Inverse(Vertex):
    def func(self, inputs):
        x = inputs[0]
        return 1. / x

    def grads(self, inputs):
        x = inputs[0]
        return [-1. / x / x]

class Squared(Vertex):
    def func(self, inputs):
        x = inputs[0]
        return x * x

    def grads(self, inputs):
        x = inputs[0]
        return [2. * x]

class Sigmoid(Vertex):
    def func(self, inputs):
        x = inputs[0]
        return 1. / (1. + np.exp(-x))

    def grads(self, inputs):
        sig = self.func(inputs)
        return [sig * (1. - sig)]

class Input(Vertex):
    def func(self, inputs):
        return self.value

    def grads(self, inputs):
        pass


