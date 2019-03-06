from backprop.graph import get_graph
from backprop.traverse import Traverse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from backprop.data import get_train_test_nn, chunks


if __name__ == "__main__":

    # Create the graph
    x, y, L = get_graph()

    # dL / dL = 1
    L.grad_value = 1.0

    learning_rate = 1.e-3
    traverse = Traverse(learning_rate)

    # Get the data
    x_train, x_test, y_train, y_test = get_train_test_nn()

    loss_history = []
    val_loss_history = []

    for epoch in range(20):
        if epoch % 1 == 0:
            print(epoch)

        for x_batch, y_batch in chunks(x_train, y_train, 10):
            x.value, y.value = x_batch, y_batch
            traverse.forward(L)
            traverse.backward()

        loss_history.append(L.value)

        # Calculate the Loss on test set
        x.value, y.value = x_test, y_test
        traverse.forward(L)
        val_loss_history.append(L.value)
        traverse.backward(update_weights=False)

    plt.figure(figsize=(4, 8))
    plt.subplot(2, 1, 1)

    plt.plot(loss_history)
    plt.plot(val_loss_history)

    # get prediction on test set
    x.value, y.value = x_test, y_test
    traverse.forward(L)
    y_pred = L.edges[0].value
    traverse.backward(update_weights=False)

    plt.subplot(2, 1, 2)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred[:, 0])
    plt.show()
