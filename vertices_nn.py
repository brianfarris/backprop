import numpy as np
from vertices_general import Vertex


class Dot(Vertex):
    def func(self, inputs):
        x, w = inputs
        return np.dot(x, w)

    def grads(self, inputs, d):
        x, w = inputs
        return [np.dot(d, w.T), np.dot(x.T, d)]


class Relu(Vertex):
    def func(self, inputs):
        x = inputs[0]
        return x * (x > 0.)

    def grads(self, inputs, d):
        x = inputs[0]
        return d * (x > 0.)


class Softmax(Vertex):
    def func(self, inputs):
        h = inputs[0]
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
        y_pred_clipped = np.clip(y_pred, 1.e-7, 1. - 1.e-7)
        return - (y_true * np.log(y_pred_clipped)).sum() / nrows

    def grads(self, inputs, d):
        y_pred, y_true = inputs
        nrows = y_true.shape[0]
        return [- y_true / y_pred / nrows,
                - np.log(y_pred) / nrows]


class AddBias(Vertex):
    def func(self, inputs):
        x, b = inputs
        return x + b

    def grads(self, inputs, d):
        x, b = inputs
        return [np.ones(x.shape) * d, np.ones(b.shape) * d.sum(axis=0)]
