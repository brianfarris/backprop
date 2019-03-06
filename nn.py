import numpy as np
from vertices_general import Input
from vertices_nn import Dot, Softmax, CrossEntropy, AddBias, Relu
from backprop import BackProp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    x = Input(name="x")
    y = Input(name="y")

    w1 = Input(name="w1", trainable=True)
    w1.value = np.random.normal(0, 0.0001, (2, 4))
    b1 = Input(name="b1", trainable=True)
    b1.value = np.zeros((1, 4))

    dot1 = Dot([x, w1], name="dot1")
    h1 = AddBias([dot1, b1], name="add1")
    h1_relu = Relu([h1], name="relu1")

    w2 = Input(name="w2", trainable=True)
    w2.value = np.diag(np.random.normal(1, 0.0001, 4))
    b2 = Input(name="b2", trainable=True)
    b2.value = np.zeros((1, 4))

    dot2 = Dot([h1_relu, w2], name="dot2")
    h2 = AddBias([dot2, b2], name="add2")
    h2_relu = Relu([h2], name="relu2")

    w3 = Input(name="w3", trainable=True)
    w3.value = np.random.normal(0, 0.0001, (4, 2))
    b3 = Input(name="b3", trainable=True)
    b3.value = np.zeros((1, 2))

    dot3 = Dot([h2_relu, w3], name="dot3")
    h3 = AddBias([dot3, b3], name="add3")

    p = Softmax([h3], name="softmax")
    L = CrossEntropy([p, y], name="cross_entropy")
    L.grad_value = 1.0

    backprop = BackProp(x, y)

    N = int(1e5)
    x_in = np.random.normal(0, 1, (N//2, 2))
    r = np.random.normal(6,0.5, (N//2, 1))
    phi = np.random.uniform(0, 2. * np.pi, (N//2, 1))
    x_out = np.hstack([r * np.cos(phi), r * np.sin(phi)])
    x_full = np.vstack([x_in, x_out])
    y_in = np.vstack([np.zeros((N//2, 1)), np.ones((N//2, 1))])
    y_out = np.vstack([np.ones((N//2, 1)), np.zeros((N//2, 1))])
    y_full = np.hstack([y_in, y_out])
    x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.1)

    loss_history = []
    val_loss_history = []
    lr = 1.e-3
    for epoch in range(20):
        if epoch % 1 == 0:
            print(epoch)
        for i in range(x_train.shape[0] // 10):
            x_batch = x_train[i * 10: (i+1)*10, :]
            y_batch = y_train[i * 10: (i+1)*10, :]

            backprop.x.value = x_batch
            backprop.y.value = y_batch

            backprop.forward(L)
            backprop.backward(lr)

        loss_history.append(L.value)

        backprop.x.value = x_test
        backprop.y.value = y_test
        backprop.forward(L)
        val_loss_history.append(L.value)
        backprop.backward(lr)

    plt.figure(figsize=(4, 8))
    plt.subplot(2, 1, 1)

    plt.plot(loss_history)
    plt.plot(val_loss_history)

    backprop.x.value = x_test
    backprop.y.value = y_test
    backprop.forward(L)
    y_pred = L.edges[0].value

    plt.subplot(2, 1, 2)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred[:, 0])
    #plt.xlim(-3, 3)
    #plt.ylim(-3, 3)
    plt.show()
