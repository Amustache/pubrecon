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
    return ((y - tx @ w) ** 2).mean()