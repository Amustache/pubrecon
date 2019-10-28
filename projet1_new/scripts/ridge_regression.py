# -*- coding: utf-8 -*-
"""Ridge Regression"""
import numpy as np


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression.

    :param y: Labels.
    :param tx: Features.
    :param lambda_: Corresponding lambda.
    :return: `w` the corresponding weights.
    """
    aI = lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)
