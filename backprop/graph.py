from vertices.vertices_nn import Dot, Softmax, CrossEntropy, AddBias, Relu
from vertices.vertices_general import Input
import numpy as np


def get_graph():
    x = Input(name="x")
    y = Input(name="y")

    # Weights for 1st hidden layer
    w1 = Input(name="w1", trainable=True)
    w1.value = np.random.normal(0, 0.0001, (2, 4))
    b1 = Input(name="b1", trainable=True)
    b1.value = np.zeros((1, 4))

    # Apply 1st hidden layer
    dot1 = Dot([x, w1], name="dot1")
    h1 = AddBias([dot1, b1], name="add1")
    h1_relu = Relu([h1], name="relu1")

    # Weights for 2nd hidden layer
    w2 = Input(name="w2", trainable=True)
    w2.value = np.diag(np.random.normal(1, 0.0001, 4))
    b2 = Input(name="b2", trainable=True)
    b2.value = np.zeros((1, 4))

    # Apply 2nd hidden layer
    dot2 = Dot([h1_relu, w2], name="dot2")
    h2 = AddBias([dot2, b2], name="add2")
    h2_relu = Relu([h2], name="relu2")

    # Weights for final dense layer
    w3 = Input(name="w3", trainable=True)
    w3.value = np.random.normal(0, 0.0001, (4, 2))
    b3 = Input(name="b3", trainable=True)
    b3.value = np.zeros((1, 2))

    # Apply final dense layer
    dot3 = Dot([h2_relu, w3], name="dot3")
    h3 = AddBias([dot3, b3], name="add3")

    # Apply softmax
    p = Softmax([h3], name="softmax")

    # use categorical crossentropy for loss
    L = CrossEntropy([p, y], name="cross_entropy")

    return x, y, L
