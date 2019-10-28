# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""
import numpy as np


def build_poly(x, degree):
    """
    Polynomial basis functions for input data x, for j=0 up to j=degree.

    :param x: Input data.
    :param degree: The corresponding degree.
    :return: The corresponding polynomial.
    """
    p = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        p = np.c_[p, np.power(x, deg)]
    return p
