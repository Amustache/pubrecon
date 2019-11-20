# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""
from .helpers import batch_iter
from .costs import compute_loss


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels.

    :param y: Labels.
    :param tx: Features.
    :param w: Weights.
    :return: `(grad, error)`, with `grad` the gradient, and `error` the corresponding error.
    """
    error = y - tx.dot(w)
    grad = -tx.T.dot(error) / len(error)
    return grad, error


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """
    Stochastic gradient descent algorithm.

    :param y: Predictions.
    :param tx: Target.
    :param initial_w: Initial weight vector.
    :param batch_size: Size of the batch.
    :param max_iters: Number of steps to run.
    :param gamma: Step-size.
    :return: `(losses, ws)`, with `ws` the weight vectors of the method, and `losses` the corresponding loss values (cost function).
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, error = compute_stoch_gradient(y_batch, tx_batch, w)
            loss = compute_loss(y, tx, w)

            # update w through the stochastic gradient update
            w = w - gamma * grad

            # store w and loss
            ws.append(w)
            losses.append(loss)

        print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws
