import numpy as np
from vertices import Multiplication, Addition, Inverse, Squared, Sigmoid, Input


def forward(vertex):
    if vertex.value is None:
        inputs = [forward(edge).value for edge in vertex.edges]
        vertex.value = vertex.func(inputs)
        stack.append(vertex)
    return vertex

def backward(stack):
    while stack:
        vertex = stack.pop()
        inputs = [edge.value for edge in vertex.edges]
        grads = vertex.grads(inputs)
        for i, edge in enumerate(vertex.edges):
            edge.grad_value += vertex.grad_value * grads[i]


if __name__ == "__main__":
    x = Input(name="x")
    x.value = np.array(3)

    y = Input(name="y")
    y.value = np.array(7)

    sigy = Sigmoid([y], name="sigy")
    numerator = Addition([x, sigy], name="numerator")
    x_plus_y = Addition([x, y], name="x_plus_y")
    x_plus_y_sq = Squared([x_plus_y], name="x_plus_y_sq")
    sigx = Sigmoid([x], name="sigx")
    denominator = Addition([sigx, x_plus_y_sq], name="denomiinator")
    denominator_inv = Inverse([denominator], name="denominator_inv")
    mult = Multiplication([numerator, denominator_inv])

    stack = []
    forward(mult)
    print("L: ", stack[-1].value)

    mult.grad_value = 1.

    backward(stack)
    print("dx: ", x.grad_value)
    print("dy: ", y.grad_value)
