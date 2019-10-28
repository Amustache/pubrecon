from sys import argv
from scripts.proj1_helpers import load_csv_data, create_csv_submission, predict_labels
from scripts.helpers import standardize, sigmoid_array
from scripts.ridge_regression import ridge_regression
from scripts.stochastic_gradient_descent import stochastic_gradient_descent
from scripts.kfold import kfold
from scripts.metrics import metrics
import numpy as np
from datetime import datetime


def model(y, tx, ids, debug=False):
    # Data preprocessing
    print("Data preprocessing...")
    print("-" * 10)

    # Standardization
    data, tx_mean, tx_std = standardize(tX)

    # Replacing odd (outliers) values with mean, column-wize
    for i in range(data.shape[1]):
        data[:, i][np.where(data[:, i] == -999)] = tx_mean[i]

    # Learning
    print("Learning...")
    print("-" * 10)
    F1_best = accuracy_best = 0
    lmbda_best = 0

    # Tuning
    for lmbda in np.arange(0, 100, 1):
        print("*" * 10)
        print("Lambda is {}.".format(lmbda))
        w = ridge_regression(y, tx, lambda_=lmbda)

        # Validation
        y_pred = predict_labels(w, tx)
        F1, accuracy = metrics(y, y_pred)
        print("F1-score: {}, accuracy: {}.".format(F1, accuracy))

        if F1 >= F1_best and accuracy >= accuracy_best:
            F1_best = F1
            accuracy_best = accuracy
            lmbda_best = lmbda

    print("Highest scoring: {}. Reducing granularity.".format(lmbda_best))

    index = lmbda_best

    for lmbda in np.arange(index - 1, index + 1, 0.01):
        print("*" * 10)
        print("Lambda is {}.".format(lmbda))
        w = ridge_regression(y, tx, lambda_=lmbda)

        # Validation
        y_pred = predict_labels(w, tx)
        F1, accuracy = metrics(y, y_pred)
        print("F1-score: {}, accuracy: {}.".format(F1, accuracy))

        if F1 >= F1_best and accuracy >= accuracy_best:
            F1_best = F1
            accuracy_best = accuracy
            lmbda_best = lmbda

    print("Highest scoring: {}.".format(lmbda_best))

    # Saving weights
    print("Saving weights...")
    print("-" * 10)
    np.save('./w {}'.format(str(datetime.now())), w)

    if debug:
        return np.random.random(30)
    else:
        return w


if __name__ == "__main__":
    if len(argv) != 4:
        raise ValueError("Error: Not enough arguments.\nUse: `python run.py <input_train_data>.csv <input_test_data>.csv <output_data>.csv`")

    if argv[1][-4:] != '.csv' or argv[2][-4:] != '.csv' or argv[3][-4:] != '.csv':
        raise ValueError("Error: Wrong filename.\nUse: `python run.py <input_train_data>.csv <input_test_data>.csv <output_data>.csv`")

    name, train_path, test_path, output_path = argv

    # Import data
    print("Importing data from {}.".format(train_path))
    y, tX, ids = load_csv_data(train_path)

    # Generate model
    print("Generating model:")
    weights = model(y, tX, ids)

    # Generating predictions
    print ("Generating predictions for {}...".format(test_path))
    print("-" * 10)
    # _, tX_test, ids_test = load_csv_data(test_path)  #  Not working somehow
    DATA_TEST_PATH = './data/test.csv'  # TODO: download train data and supply path here
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

    y_pred = predict_labels(weights, tX_test)

    # Export predictions
    print("Exporting predictions to {}.".format(output_path))
    print("-" * 10)
    # create_csv_submission(ids, y_pred, output_path)
    OUTPUT_PATH = './data/output.csv'  # TODO: fill in desired name of output file for submission
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
