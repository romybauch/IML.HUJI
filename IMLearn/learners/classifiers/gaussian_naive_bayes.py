from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in
            `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in
            `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in
            `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        n_samples = X.shape[0]
        n_classes = self.classes_.shape[0]

        # initialize mu,cov,pi matrix
        self.mu_ = np.zeros((n_classes, n_features))
        self.vars_ = np.zeros((n_classes, n_features))
        self.pi_ = np.zeros(n_classes)

        for i, k in enumerate(self.classes_):
            x_k = X[y == k]
            self.mu_[i] = np.mean(x_k, axis=0)
            self.pi_[i] = x_k.shape[0] / n_samples
            self.vars_[i] = np.var(x_k, axis=0)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        all_pred = np.zeros((self.classes_.shape[0], X.shape[0]))  # n_cXn_s
        for i in range(self.classes_.shape[0]):
            x_mu = X - self.mu_[i]
            log_pi_i = np.log(self.pi_[i])
            sum_i = np.sum((x_mu**2)/self.vars_[i] + np.log(self.vars_[i]),
                           axis=1)
            all_pred[i] = log_pi_i -0.5*sum_i  # n_samples

        max_pred = np.argmax(all_pred, axis=0)  # n_samples

        label = lambda i: self.classes_[i]
        return np.fromiter(map(label, max_pred), dtype=type(self.classes_[0]))

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        # if not self.fitted_:
        #     raise ValueError("Estimator must first be fitted before calling "
        #                      "`likelihood` function")
        #
        # likelihood = np.zeros((self.classes_.shape[0],X.shape[0])) # classXsample
        # for i in range(self.classes_.shape[0]):
        #     pi_class = self.pi_[i]
        #     var_class = self.vars_[i]
        #     x_m = X - self.mu_[i]
        #     likelihood[i] = pi_class*np.product(np.exp(-(x_m)**2/2*var_class)/
        #                                         np.square(2*np.pi*var_class))
        #
        # return likelihood.T #sampleXclass
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling "
                             "`likelihood` function")
        # log likelihood calculate
        loglikely = np.zeros((X.shape[0], self.classes_.shape[0]))
        for i in range(self.classes_.shape[0]):
            x_mu = (X - self.mu_[i]) ** 2
            sub_sum = self.vars_[i] + np.log(self.vars_[i])
            log_pi = np.log(self.pi_[i])
            sum_i = 1 / 2 * np.sum(((x_mu) / sub_sum), axis=1)
            loglikely[:, i] = log_pi - sum_i
        return loglikely

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
        from ...metrics import misclassification_error
        return misclassification_error(self.predict(X), y)
