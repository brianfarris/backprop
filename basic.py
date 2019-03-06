import numpy as np
from vertices_general import Input
from vertices_basic import Multiplication, Addition, Inverse, Squared, Sigmoid
from backprop import BackProp


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

    backprop = BackProp(x, y)
    backprop.forward(mult)
    print("L: ", backprop.stack[-1].value)

    mult.grad_value = 1.

    backprop.backward(1.0, print_grads=True)
