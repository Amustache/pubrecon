import numpy as np


def sigmoid(x):
    """
    Sigmoid activation function.
    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def next_batch(X, tx, batch_size):
    """
    Yields mini-batches of data.
    :param X: Dataset.
    :param tx: Labels.
    :param batch_size: Size of each mini-batch.
    :return: Mini-batch.
    """
    for i in np.arange(0, X.shape[0], batch_size):
        yield (X[i:i + batch_size], tx[i:i + batch_size])


def gradient(y, tx, initial_w):
    est = y.dot(initial_w)
    error = tx - est
    grad = -(1. / len(tx)) * error.dot(tx)
    return grad, np.pow(error, 2)


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent.
    :param y: Predictions.
    :param tx: Target.
    :param initial_w: Initial weight vector.
    :param max_iters: Number of steps to run.
    :param gamma: Step-size.
    :return: `(w, loss)`, with `w` the last weight vector of the method, and `loss` the corresponding loss value (cost function).
    """
    w = initial_w
    loss = 0
    for i in range(max_iters):
        grad, loss = gradient(y, tx, w)
        w = w - gamma * grad
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent.
    :param y: Predictions.
    :param tx: Target.
    :param initial_w: Initial weight vector.
    :param max_iters: Number of steps to run.
    :param gamma: Step-size.
    :return: `(w, loss)`, with `w` the last weight vector of the method, and `loss` the corresponding loss value (cost function).
    """


def least_squares(y, tx):
    """
    Least squares regression using normal equations.
    :param y: Predictions.
    :param tx: Target.
    :return: `(w, loss)`, with `w` the last weight vector of the method, and `loss` the corresponding loss value (cost function).
    """


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations.
    :param y: Predictions.
    :param tx: Target.
    :param lambda_: Regularization parameter.
    :return: `(w, loss)`, with `w` the last weight vector of the method, and `loss` the corresponding loss value (cost function).
    """


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent or SGD.
    :param y: Predictions.
    :param tx: Target.
    :param initial_w: Initial weight vector.
    :param max_iters: Number of steps to run.
    :param gamma: Step-size.
    :return: `(w, loss)`, with `w` the last weight vector of the method, and `loss` the corresponding loss value (cost function).
    """


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent or SGD.
    :param y: Predictions.
    :param tx: Target.
    :param lambda_: Regularization parameter.
    :param initial_w: Initial weight vector.
    :param max_iters: Number of steps to run.
    :param gamma: Step-size.
    :return: `(w, loss)`, with `w` the last weight vector of the method, and `loss` the corresponding loss value (cost function).
    """
