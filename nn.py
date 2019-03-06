import numpy as np
from vertices_general import Input
from vertices_nn import Dot, Softmax, CrossEntropy, AddBias, Relu
from backprop import BackProp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data import get_train_test, chunks


if __name__ == "__main__":
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

    # dL / dL = 1
    L.grad_value = 1.0

    learning_rate = 1.e-3
    backprop = BackProp(learning_rate)

    # Get the data
    x_full, y_full = get_train_test()
    (x_train, x_test,
     y_train, y_test) = train_test_split(x_full, y_full,
                                         test_size=0.1)

    loss_history = []
    val_loss_history = []

    for epoch in range(20):
        if epoch % 1 == 0:
            print(epoch)

        for x_batch, y_batch in chunks(x_train, y_train, 10):
            x.value = x_batch
            y.value = y_batch

            backprop.forward(L)
            backprop.backward()

        loss_history.append(L.value)

        # Calculate the Loss on test set
        x.value, y.value = x_test, y_test
        backprop.forward(L)
        val_loss_history.append(L.value)
        backprop.backward()

    plt.figure(figsize=(4, 8))
    plt.subplot(2, 1, 1)

    plt.plot(loss_history)
    plt.plot(val_loss_history)

    # get prediction on test set
    x.value, y.value = x_test, y_test
    backprop.forward(L)
    y_pred = L.edges[0].value
    backprop.backward()

    plt.subplot(2, 1, 2)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred[:, 0])
    plt.show()
