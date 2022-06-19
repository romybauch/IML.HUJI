from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float],
                   cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated
        model. When called, the scoring function receives the true- and
        predicted values for each sample and potentially additional arguments.
        The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    error_train, error_val = 0, 0
    sep = np.remainder(np.arange(y.size), cv) # 0,1,..,cv,0,1,...,cv...
    for i in range(cv):
        x_train_i , y_train_i = X[sep!=i], y[sep!=i]
        x_val_i, y_val_i = X[sep==i], y[sep==i]

        estimator.fit(x_train_i, y_train_i)
        error_train += scoring(estimator.predict(x_train_i), y_train_i)
        error_val += scoring(estimator.predict(x_val_i), y_val_i)

    return error_train/cv, error_val/cv






