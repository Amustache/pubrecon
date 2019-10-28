# -*- coding: utf-8 -*-
"""Compulsory implementations"""
from numpy import np
from .costs import compute_loss
from .helpers import batch_iter, sigmoid
from .stochastic_gradient_descent import compute_stoch_gradient
from .gradient_descent import compute_gradient


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent.
    :param y: Labels.
    :param tx: Features.
    :param initial_w: Initial weight vector.
    :param max_iters: Number of steps to run.
    :param gamma: Step-size.
    :return: `(w, loss)`, with `w` the last weight vector of the method, and `loss` the corresponding loss value (cost function).
    """
    w = initial_w

    for n_iter in range(max_iters):
        # compute gradient and loss
        grad, error = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)

        # update w by gradient
        w = w - gamma * grad

        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss,
                                                                               w0=w[0], w1=w[1]))

    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent.
    :param y: Labels.
    :param tx: Features.
    :param initial_w: Initial weight vector.
    :param max_iters: Number of steps to run.
    :param gamma: Step-size.
    :return: `(w, loss)`, with `w` the last weight vector of the method, and `loss` the corresponding loss value (cost function).
    """
    # Define parameters to store w and loss
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx):
            # compute a stochastic gradient and loss
            grad, error = compute_stoch_gradient(y_batch, tx_batch, w)
            loss = compute_loss(y, tx, w)

            # update w through the stochastic gradient update
            w = w - gamma * grad

        print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return w, loss


def least_squares(y, tx):
    """
    Least squares regression using normal equations.
    :param y: Labels.
    :param tx: Features.
    :return: `(w, loss)`, with `w` the last weight vector of the method, and `loss` the corresponding loss value (cost function).
    """
    w, res, rank, s = np.linalg.lstsq(tx, y)
    loss = compute_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations.
    :param y: Labels.
    :param tx: Features.
    :param lambda_: Regularization parameter.
    :return: `(w, loss)`, with `w` the last weight vector of the method, and `loss` the corresponding loss value (cost function).
    """
    aI = lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent or SGD.
    :param y: Labels.
    :param tx: Features.
    :param initial_w: Initial weight vector.
    :param max_iters: Number of steps to run.
    :param gamma: Step-size.
    :return: `(w, loss)`, with `w` the last weight vector of the method, and `loss` the corresponding loss value (cost function).
    """
    w = initial_w
    for n_iter in range(max_iters):
        # computer gradient and loss
        grad, error = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)

        # update w by gradient
        w = w - gamma * grad

        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss,
                                                                               w0=w[0], w1=w[1]))
        y_pred = sigmoid(np.dot(tx, w))

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Rfregression using gradient descent or SGD.
    :param y: Labels.
    :param tx: Features.
    :param lambda_: Regularization parameter.
    :param initial_w: Initial weight vector.
    :param max_iters: Number of steps to run.
    :param gamma: Step-size.
    :return: `(w, loss)`, with `w` the last weight vector of the method, and `loss` the corresponding loss value (cost function).
    """
    w = initial_w
