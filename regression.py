import numpy as np
from vertices import Input, Dot, Softmax, CrossEntropy, AddBias
from backprop import BackProp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    x = Input(name="x")
    y = Input(name="y")
    w = Input(name="w", trainable=True)
    w.value = np.random.normal(0, 0.001, (2, 2))
    b = Input(name="b", trainable=True)
    b.value = np.random.normal(0, 0.001, (1, 2))

    dot = Dot([x, w], name="dot")
    h = AddBias([dot, b], name="add")
    p = Softmax([h], name="softmax")
    L = CrossEntropy([p, y], name="cross_entropy")

    backprop = BackProp(x, y, L, 1.)

    N = int(1e3)
    np.random.seed(42)
    x_full = np.random.normal(0, 1, (N, 2))
    noise = np.random.normal(0, .01, N)
    y_full = np.hstack([(x_full[:, 0] + x_full[:, 1] > noise).astype(int).reshape(N, 1),
        (x_full[:, 0] + x_full[:, 1] < noise).astype(int).reshape(N, 1)])
    x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.1)

    loss_history = []
    val_loss_history = []
    for epoch in range(100):
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
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.show()
