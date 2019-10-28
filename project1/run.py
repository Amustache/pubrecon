from sys import argv
import numpy as np
from script.proj1_helpers import load_csv_data, create_csv_submission, standardize, predict_labels
from script.gradient_descent import gradient_descent

def kfold(y, tx, ids, folds=10, shuffle=True):
    """
    K-fold for testing.

    :param y: Labels.
    :param tx: Features.
    :param ids: Corresponding IDs.
    :param folds: Number of folds.
    :param shuffle: Should we shuffle the data prior to the fold?
    :return: K-fold iterator, test_set, validation_set
    """
    data_size = len(y)
    fold_size = int(data_size / folds)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
        shuffled_ids = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
        shuffled_ids = ids

    for fold_num in range(folds):
        start = fold_num * fold_size
        end = min((fold_num + 1) * fold_size, data_size)

        if start != end:
            test_y, validation_y = np.concatenate((shuffled_y[:start], shuffled_y[end:])), shuffled_y[start:end]
            test_tx, validation_tx = np.concatenate((shuffled_tx[:start], shuffled_tx[end:])), shuffled_tx[start:end]
            test_ids, validation_ids = np.concatenate((shuffled_ids[:start], shuffled_ids[end:])), shuffled_ids[start:end]
            yield test_y, test_tx, test_ids, validation_y, validation_tx, validation_ids


def model(y, tX, ids, epochs=7):
    # Standardize data
    tX, mean_tX, std_tX = standardize(tX)

    # # We will search for the hyperparameters
    # for test_y, test_tX, test_ids, validation_y, validation_tX, validation_ids in kfold(y, tX, ids):
    #     pass

    loss, w = gradient_descent(y, tX, np.random.random(y.shape[0]), epochs, np.random.randint(1))

    y_pred = predict_labels(w, tX_test)

    return ids, y_pred


if __name__ == "__main__":
    if len(argv) != 3:
        raise ValueError("Error: Not enough arguments.\nUse: `python run.py <input_data>.csv <output_data>.csv`")

    if argv[1][-4:] != '.csv' or argv[2][-4:] != '.csv':
        raise ValueError("Error: Wrong filename.\nUse: `python run.py <input_data>.csv <output_data>.csv`")

    name, inpt, outp = argv

    # Import data
    print("Importing data from {}.".format(inpt))
    y, tX, ids = load_csv_data(inpt, sub_sample=True)

    # Generate predictions
    print("Generating predictions.")
    ids, y_pred = model(y, tX, ids)

    # Export predictions
    print("Exporting predictions to {}.".format(outp))
    create_csv_submission(ids, y_pred, outp)