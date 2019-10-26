# -*- coding: utf-8 -*-
"""Gradient Descent"""

def compute_gradient(y, tx, w):
    """
    Compute the gradient.

    :param y: Labels.
    :param tx: Features.
    :param w: Weights.
    :return: `(w, loss)`, with `w` the last weight vector of the method, and `loss` the corresponding loss value (cost function).
    """
    error = compute_loss(y, tx, w)
    return -(1. / tx.shape[0]) * tx.T.dot(error), error


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """
    Gradient descent algorithm.

    :param y: Labels.
    :param tx: Features.
    :param initial_w: Initial weight vector.
    :param max_iters: Number of steps to run.
    :param gamma: Step-size.
    :return:`(w, loss)`, with `w` the last weight vector of the method, and `loss` the corresponding loss value (cost function).
    """
    # Define parameters to store w and loss
    w = initial_w
    loss = 0
    for n_iter in range(max_iters):
        # compute gradient and loss
        grad, loss = compute_gradient(y, tx, w)

        # update w by gradient
        w = w - gamma * grad

    return loss, w
