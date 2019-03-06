import numpy as np


class Vertex:
    def __init__(self, edges=[], name=None, trainable=False):
        self.name = name
        self.edges = edges
        self.value = None
        self.grad_value = 0
        self.trainable = trainable
        self.visited = None


class Input(Vertex):
    def func(self, inputs):
        return self.value

    def grads(self, inputs, d):
        pass
