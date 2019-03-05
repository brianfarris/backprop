import numpy as np


class BackProp:
    def __init__(self, x, y, loss, step_size):
        self.stack = []
        self.x = x
        self.y = y
        self.loss = loss
        self.loss.grad_value = 1.0
        self.step_size = step_size

    def forward(self, vertex):
        inputs = [self.forward(edge).value for edge in vertex.edges]
        vertex.value = vertex.func(inputs)
        self.stack.append(vertex)
        return vertex

    def backward(self):
        while self.stack:
            vertex = self.stack.pop()
            inputs = [edge.value for edge in vertex.edges]
            grads = vertex.grads(inputs, vertex.grad_value)
            for i, edge in enumerate(vertex.edges):
                edge.grad_value += grads[i]
            if vertex.trainable:
                vertex.value -= (self.step_size * vertex.grad_value)
            vertex.grad_value = 0.0

    def batch(self, x_batch, y_batch):
        self.x.value = x_batch
        self.y.value = y_batch
        self.forward(self.loss)
        self.backward()

    def predict(self, x, y):
        self.x.value = x
        self.y.value = y
        self.forward(self.loss)
        return self.loss.edges[0].value
