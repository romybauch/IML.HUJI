from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = \
            None, None, None, None, None
        self.a,self.b = None,None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same
        covariance matrix with dependent features.

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
        self.mu_ = np.zeros((n_classes,n_features))
        self.cov_ = np.zeros((n_features,n_features))
        self.pi_ = np.zeros(n_classes)

        for i,k in enumerate(self.classes_):
            x_k = X[y == k]
            self.mu_[i] = np.mean(x_k, axis=0)
            self.pi_[i] = x_k.shape[0]/n_samples
            self.cov_ += (x_k-self.mu_[i]).T.dot(x_k-self.mu_[i])

        self.cov_ /= n_samples-n_classes
        self._cov_inv = np.linalg.inv(self.cov_)
        self.a = self._cov_inv@self.mu_.T # n_feature X n_class
        self.b = (-0.5 * np.einsum('ij,ij->i', self.mu_ @ self._cov_inv,
                                   self.mu_) + np.log(self.pi_)) # n_class

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
        a_t = self.a.T # n_class X n_features
        all_pred = np.zeros((self.classes_.shape[0],X.shape[0])) # n_cXn_s
        for i in range(self.classes_.shape[0]):
            a_i = a_t[i] # n_feature
            b_i = self.b[i] # 1
            all_pred[i] = a_i@X.T+b_i # n_samples

        max_pred = np.argmax(all_pred, axis=0) # n_samples

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
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling "
                             "`likelihood` function")
        likelihoods = np.zeros((X.shape[0],self.classes_.shape[0])) # n_sXn_c
        det_cov = np.linalg.det(self.cov_)
        z = np.square(det_cov * (2 * np.pi) ** X.shape[1])

        for i in range(self.classes_.shape[0]):
            for j,sample in enumerate(X):
                xi = sample - self.mu_[i]
                k = self.pi_[i]*(1/z)*np.exp(-0.5*xi@self._cov_inv@xi.T)
                likelihoods[j,i] = k

        return likelihoods

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
        return misclassification_error(self.predict(X),y)
