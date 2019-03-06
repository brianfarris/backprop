import numpy as np
from vertices.vertices_general import Input
from vertices.vertices_basic import Multiplication, Addition, Inverse, Squared, Sigmoid
from backprop.traverse import Traverse


if __name__ == "__main__":
    x = Input(name="x")
    x.value = np.array(3)

    y = Input(name="y")
    y.value = np.array(-4)

    sigy = Sigmoid([y], name="sigy")
    numerator = Addition([x, sigy], name="numerator")
    x_plus_y = Addition([x, y], name="x_plus_y")
    x_plus_y_sq = Squared([x_plus_y], name="x_plus_y_sq")
    sigx = Sigmoid([x], name="sigx")
    denominator = Addition([sigx, x_plus_y_sq], name="denominator")
    denominator_inv = Inverse([denominator], name="denominator_inv")
    mult = Multiplication([numerator, denominator_inv])

    learning_rate = 1.0
    traverse = Traverse(learning_rate)
    traverse.forward(mult)
    print("L: ", traverse.stack[-1].value)

    mult.grad_value = 1.

    traverse.backward(print_grads=True)
