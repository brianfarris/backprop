import numpy as np
from vertices import Multiplication, Addition, Inverse, Squared, Sigmoid, Input
from backprop import BackProp


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
    denominator = Addition([sigx, x_plus_y_sq], name="denominator")
    denominator_inv = Inverse([denominator], name="denominator_inv")
    mult = Multiplication([numerator, denominator_inv])


    backprop = BackProp()
    backprop.forward(mult)
    print("L: ", backprop.stack[-1].value)

    mult.grad_value = 1.

    backprop.backward()
    print("dx: ", x.grad_value)
    print("dy: ", y.grad_value)
