import numpy as np


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
