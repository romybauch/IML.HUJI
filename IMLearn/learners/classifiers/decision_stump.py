from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product
from IMLearn.metrics import loss_functions


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        loss_min = np.inf
        X_T = X.T
        for sign, feature_ind in product([-1,1], range(X_T.shape[0])):
            t_thresh, t_loss = \
                self._find_threshold(X_T[feature_ind], y, sign)
            if t_loss < loss_min:
                loss_min = t_loss
                self.sign_ = sign
                self.j_ = feature_ind
                self.threshold_ = t_thresh

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign`
        whereas values which equal to or above the threshold are predicted as
        `sign`
        """
        y_hat = np.ones(X.shape[0])*self.sign_
        y_hat[X[:, self.j_] < self.threshold_] = -self.sign_
        return y_hat

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray,
                        sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform
        a split The threshold is found according to the value minimizing the
        misclassification error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are
        predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        pred_y = np.ones(values.shape[0])*sign
        sorted_idx = np.argsort(values)
        value_sorted, label_sorted = values[sorted_idx], labels[sorted_idx]

        threshold = values[0]
        min_err = np.inf
        for i, thresh_val in enumerate(value_sorted):
            err = np.sum(np.abs(label_sorted*(np.sign(label_sorted) != pred_y)))
            if err <= min_err:
                min_err, threshold = err, thresh_val
            pred_y[i] = -sign

        return threshold, min_err

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_true = np.where(y >= 0, 1, -1)
        return loss_functions.misclassification_error(self.predict(X) ,y_true)

