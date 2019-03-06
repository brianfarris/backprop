import numpy as np
from vertices.vertices_general import Vertex


class Multiplication(Vertex):
    def func(self, inputs):
        x, y = inputs
        return x * y

    def grads(self, inputs, d):
        x, y = inputs
        return [y * d, x * d]


class Addition(Vertex):
    def func(self, inputs):
        x, b = inputs
        return x + b

    def grads(self, inputs, d):
        x, b = inputs
        return [np.ones(x.shape) * d, np.ones(b.shape) * d]


class Inverse(Vertex):
    def func(self, inputs):
        x = inputs[0]
        return 1. / x

    def grads(self, inputs, d):
        x = inputs[0]
        return [-1. / x / x * d]


class Squared(Vertex):
    def func(self, inputs):
        x = inputs[0]
        return x * x

    def grads(self, inputs, d):
        x = inputs[0]
        return [2. * x * d]


class Sigmoid(Vertex):
    def func(self, inputs):
        x = inputs[0]
        return 1. / (1. + np.exp(-x))

    def grads(self, inputs, d):
        sig = self.func(inputs)
        return [sig * (1. - sig) * d]
