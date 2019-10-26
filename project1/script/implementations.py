import numpy as np


# Compulsory implementations
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
        est = y.dot(initial_w)
        error = tx - est
        grad = (1. / tx.shape[0]) * error.dot(tx)
        loss = loss_function(tx, est)
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
    w = initial_w
    loss = 0
    for i in range(max_iters):
        for (batch_x, batch_tx) in next_batch(y, tx):
            est = batch_x.dot(initial_w)
            error = batch_tx - est
            grad = (1. / batch_x.shape[0]) * batch_x.dot(error)
            loss = loss_function(batch_tx, est)
            w = w - gamma * grad
    return w, loss


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
