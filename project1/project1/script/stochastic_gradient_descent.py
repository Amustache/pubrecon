# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""
from proj1_helpers import compute_loss, batch_iter


def compute_stoch_gradient(y, tx, w):
    """
    Compute a stochastic gradient from just few examples n and their corresponding y_n labels.

    :param y: Labels.
    :param tx: Features.
    :param w: Weights.
    :return: `(w, loss)`, with `w` the last weight vector of the method, and `loss` the corresponding loss value (cost function).
    """
    error = compute_loss(y, tx, w)
    return -(1. / tx.shape[0]) * tx.T.dot(error), error


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """
    Stochastic gradient descent algorithm.

    :param y: Predictions.
    :param tx: Target.
    :param initial_w: Initial weight vector.
    :param batch_size: Size of the batch.
    :param max_iters: Number of steps to run.
    :param gamma: Step-size.
    :return:`(w, loss)`, with `w` the last weight vector of the method, and `loss` the corresponding loss value (cost function).
    """
    w = initial_w
    loss = 0
    for i in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, num_batches=batch_size):
            # compute gradient and loss
            grad, loss = compute_stoch_gradient(minibatch_y, minibatch_tx, w)

            # update w by gradient
            w = w - gamma * grad
    return w, loss
