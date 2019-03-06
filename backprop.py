import numpy as np


class BackProp:
    def __init__(self, x, y):
        self.stack = []
        self.x = x
        self.y = y

    def forward(self, vertex):
        if not vertex.visited:
            inputs = [self.forward(edge).value for edge in vertex.edges]
            vertex.value = vertex.func(inputs)
            vertex.visited = True
            self.stack.append(vertex)
        return vertex

    def backward(self, step_size, print_grads=False, update_weights=True):
        while self.stack:
            vertex = self.stack.pop()
            inputs = [edge.value for edge in vertex.edges]
            grads = vertex.grads(inputs, vertex.grad_value)
            for i, edge in enumerate(vertex.edges):
                edge.grad_value += grads[i]
            if vertex.trainable and update_weights:
                vertex.value -= (step_size * vertex.grad_value)
            vertex.visited = False
            if print_grads:
                print(vertex.name, "gradient: ", vertex.grad_value)
            vertex.grad_value = 0.0
