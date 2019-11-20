# -*- coding: utf-8 -*-


def compute_loss(y, tx, w):
    """
    Calculate the loss by MSE.

    :param y: Labels.
    :param tx: Features.
    :param w: Weights.
    :return: Error, MSE-style.
    """
    return ((y - tx.dot(w)) ** 2).mean() / 2.
