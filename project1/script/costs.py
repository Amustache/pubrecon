# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

def compute_loss(y, tx, w):
    """
    Calculate the loss by MSE.

    :param y: Labels.
    :param tx: Features.
    :param w: Weights.
    :return: Error, MSE-style.
    """
    print(y.shape)
    print(tx.T.shape)
    print(w.shape)

    return ((y - tx.dot(w)) ** 2).mean()
