# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

def compute_loss(y, tx, w):
    """
    Calculate the loss.

    :param y: Labels.
    :param tx: Features.
    :param w: Weights.
    :return:
    """
    return ((y - tx @ w) ** 2).mean()