from typing import Tuple
import numpy as np
import pandas as pd


def split_train_test(X: pd.DataFrame, y: pd.Series, train_proportion: float = .75) \
        -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Randomly split given sample to a training- and testing sample

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Data frame of samples and feature values.

    y : Series of shape (n_samples, )
        Responses corresponding samples in data frame.

    train_proportion: Fraction of samples to be split as training set

    Returns
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples

    """
    # train_x = X.sample(frac=train_proportion)
    # train_y = y[train_x.index]
    #
    # test_x = X.drop(index = train_x.index)
    # test_y = y.drop(index = train_x.index)
    #
    # return pd.DataFrame(train_x), pd.Series(train_y), \
    #        pd.DataFrame(test_x), pd.Series(test_y)

    train_X = X.sample(frac=train_proportion, random_state=0)
    test_X = X.drop(train_X.index)
    train_y = y.sample(frac=train_proportion, random_state=0)
    test_y = y.drop(train_y.index)
    return train_X, train_y, test_X, test_y


def confusion_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute a confusion matrix between two sets of integer vectors

    Parameters
    ----------
    a: ndarray of shape (n_samples,)
        First vector of integers

    b: ndarray of shape (n_samples,)
        Second vector of integers

    Returns
    -------
    confusion_matrix: ndarray of shape (a_unique_values, b_unique_values)
        A confusion matrix where the value of the i,j index shows the number of times value `i` was found in vector `a`
        while value `j` vas found in vector `b`
    """
    uniq_a = np.unique(a)
    uniq_b = np.unique(b)

    conf_matrix = np.zeros((uniq_a.shape[0], uniq_b.shape[0]))

    for i in range(uniq_a.shape[0]):
        for j in range(uniq_b.shape[0]):
            conf_matrix[i,j] = np.sum((a == uniq_a[i]) & (b == uniq_b[j]))

    return conf_matrix