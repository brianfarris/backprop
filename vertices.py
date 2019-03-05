import numpy as np


class Vertex:
    def __init__(self, edges=[], name=None, trainable=False):
        self.name = name
        self.edges = edges
        self.value = None
        self.grad_value = 0
        self.trainable = trainable


class Dot(Vertex):
    def func(self, inputs):
        x, w = inputs
        return np.dot(x, w)

    def grads(self, inputs, d):
        x, w = inputs
        return [np.dot(d, w.T), np.dot(x.T, d)]


class Softmax(Vertex):
    def func(self, inputs):
        h = inputs[0]
        #print("h: ", h)
        num = np.exp(h - h.max(axis=1, keepdims=True))
        denom = num.sum(axis=1, keepdims=True)
        return num / denom

    def grads(self, inputs, d):
        f = self.func(inputs)
        return [f * d - (f * d).sum(axis=1, keepdims=True) * f]


class CrossEntropy(Vertex):
    def func(self, inputs):
        y_pred, y_true = inputs
        nrows = y_true.shape[0]
        y_pred_clipped = np.clip(y_pred, 1.e-17, 1. - 1.e-17)
        return - (y_true * np.log(y_pred_clipped)).sum() / nrows

    def grads(self, inputs, d):
        y_pred, y_true = inputs
        nrows = y_true.shape[0]
        return [- y_true / y_pred / nrows,
                - np.log(y_pred) / nrows]

class Multiplication(Vertex):
    def func(self, inputs):
        x, y = inputs
        return x * y

    def grads(self, inputs, d):
        x, y = inputs
        return [y * d, x * d]


class Addition(Vertex):
    def func(self, inputs):
        x, y = inputs
        return x + y

    def grads(self, inputs, d):
        x, y = inputs
        return [np.ones(x.shape) * d, np.ones(y.shape) * d]


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

class Input(Vertex):
    def func(self, inputs):
        return self.value

    def grads(self, inputs, d):
        pass
