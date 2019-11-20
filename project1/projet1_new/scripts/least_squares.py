# -*- coding: utf-8 -*-
"""Least Squares Solutions"""
import numpy as np


def least_squares(y, tx):
    """
    Calculate the least squares solution for the formula `tx . w = y`.
    :param y: Labels.
    :param tx: Features.
    :return: `w` the corresponding weights vector.
    """
    w, res, rank, s = np.linalg.lstsq(tx.T.dot(tx), tx.T.dot(y))
    return w
