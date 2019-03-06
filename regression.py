import numpy as np
from vertices.vertices_general import Input
from vertices.vertices_nn import Dot, Softmax, CrossEntropy, AddBias
from backprop.traverse import Traverse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from backprop.data import get_train_test_reg, chunks


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

    traverse = Traverse(1.)

    # Get the Data
    x_train, x_test, y_train, y_test = get_train_test_reg()

    loss_history = []
    val_loss_history = []
    for epoch in range(100):
        for x_batch, y_batch in chunks(x_train, y_train, 10): 
            x.value, y.value = x_batch, y_batch
            traverse.forward(L)
            traverse.backward()

        loss_history.append(L.value)

        x.value, y.value = x_test, y_test
        traverse.forward(L)
        val_loss_history.append(L.value)
        traverse.backward(update_weights=False)

    plt.figure(figsize=(4, 8))
    plt.subplot(2, 1, 1)

    plt.plot(loss_history)
    plt.plot(val_loss_history)

    x.value, y.value = x_test, y_test
    traverse.forward(L)
    y_pred = L.edges[0].value
    traverse.backward(update_weights=False)

    plt.subplot(2, 1, 2)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred[:, 0])
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.show()
