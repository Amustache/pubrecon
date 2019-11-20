# -*- coding: utf-8 -*-
"""Gradient Descent"""
from costs import compute_loss


def compute_gradient(y, tx, w):
    """
    Compute the gradient.

    :param y: Labels.
    :param tx: Features.
    :param w: Weights.
    :return: `(grad, error)`, with `grad` the gradient, and `error` the corresponding error.
    """
    error = y - tx.dot(w)
    grad = -tx.T.dot(error) / len(error)
    return grad, error


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """
     Gradient descent algorithm.

     :param y: Labels.
     :param tx: Features.
     :param initial_w: Initial weight vector.
     :param max_iters: Number of steps to run.
     :param gamma: Step-size.
     :return: `(losses, ws)`, with `ws` the weight vectors of the method, and `losses` the corresponding loss values (cost function).
     """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        # compute gradient and loss
        grad, error = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)

        # update w by gradient
        w = w - gamma * grad

        # store w and loss
        ws.append(w)
        losses.append(loss)

        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws
