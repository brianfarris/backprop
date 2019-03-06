import numpy as np
from vertices import Input, Dot, Softmax, CrossEntropy, AddBias, Relu
from backprop import BackProp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    x = Input(name="x")
    y = Input(name="y")

    w1 = Input(name="w1", trainable=True)
    w1.value = np.random.normal(0, 0.001, (2, 4))
    b1 = Input(name="b1", trainable=True)
    b1.value = np.random.normal(0, 0.001, (1, 4))

    dot1 = Dot([x, w1], name="dot1")
    h1 = AddBias([dot1, b1], name="add1")
    h1_relu = Relu([h1], name="relu1")

    w1 = Input(name="w2", trainable=True)
    w1.value = np.random.normal(0, 0.001, (4, 2))
    b1 = Input(name="b2", trainable=True)
    b1.value = np.random.normal(0, 0.001, (1, 2))

    dot1 = Dot([h1_relu, w1], name="dot2")
    h1 = AddBias([dot1, b1], name="add2")

    p = Softmax([h1], name="softmax")
    L = CrossEntropy([p, y], name="cross_entropy")

    backprop = BackProp(x, y, L, 0.001)

    N = int(1e4)
    np.random.seed(42)
    x_full = np.random.normal(0, 1, (N, 2))
    noise = np.random.normal(0, .01, N)
    y_full = (np.sqrt(x_full[:, 0]**2 + x_full[:, 1]**2) < (1+noise)).astype(int)
    y_full = np.vstack([y_full, (1-y_full)]).T
    x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.1)

    loss_history = []
    val_loss_history = []
    for epoch in range(2000):
        if epoch % 100 == 0:
            print(epoch)
        for i in range(x_train.shape[0] // 10):
            x_batch = x_train[i * 10: (i+1)*10, :]
            y_batch = y_train[i * 10: (i+1)*10, :]
            backprop.batch(x_batch, y_batch)
        loss_history.append(backprop.loss.value)
        backprop.x.value = x_test
        backprop.y.value = y_test
        backprop.forward(backprop.loss)
        val_loss_history.append(backprop.loss.value)

    plt.figure(figsize=(4, 8))
    plt.subplot(2, 1, 1)

    plt.plot(loss_history)
    plt.plot(val_loss_history)

    y_pred = backprop.predict(x_test, y_test)

    plt.subplot(2, 1, 2)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred[:, 0])
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.show()
