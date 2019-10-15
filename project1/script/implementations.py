import numpy as np

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent.
    :param y: Predictions.
    :param tx: Target. 
    :param initial_w: Initial weight vector. 
    :param max_iters: Number of steps to run. 
    :param gamma: Step-size. 
    :return: `(w, loss)`, with `w` the last weight vector of the method, and `loss` the corresponding loss value (cost function).
    """
    

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent.
    :param y: Predictions.
    :param tx: Target. 
    :param initial_w: Initial weight vector. 
    :param max_iters: Number of steps to run. 
    :param gamma: Step-size. 
    :return: `(w, loss)`, with `w` the last weight vector of the method, and `loss` the corresponding loss value (cost function).
    """
    

def least_squares(y, tx):
    """
    Least squares regression using normal equations.
    :param y: Predictions.
    :param tx: Target. 
    :return: `(w, loss)`, with `w` the last weight vector of the method, and `loss` the corresponding loss value (cost function).
    """
    

def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations.
    :param y: Predictions.
    :param tx: Target. 
    :param lambda_: Regularization parameter. 
    :return: `(w, loss)`, with `w` the last weight vector of the method, and `loss` the corresponding loss value (cost function).
    """
    

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent or SGD.
    :param y: Predictions.
    :param tx: Target.
    :param initial_w: Initial weight vector.
    :param max_iters: Number of steps to run.
    :param gamma: Step-size.
    :return: `(w, loss)`, with `w` the last weight vector of the method, and `loss` the corresponding loss value (cost function).
    """
    

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent or SGD.
    :param y: Predictions.
    :param tx: Target.
    :param lambda_: Regularization parameter.
    :param initial_w: Initial weight vector.
    :param max_iters: Number of steps to run.
    :param gamma: Step-size. 
    :return: `(w, loss)`, with `w` the last weight vector of the method, and `loss` the corresponding loss value (cost function).
    """
