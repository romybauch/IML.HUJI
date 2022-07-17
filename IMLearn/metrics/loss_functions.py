import numpy as np


def mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate MSE loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    MSE of given predictions
    """
    return float(sum(pow(y_true-y_pred,2))/len(y_true))

def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray,
                            normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not

    Returns
    -------
    Misclassification of given predictions
    """
    if y_true.size == 0 or y_pred.size == 0:
        return 0

    equal = np.zeros(y_true.shape[0])
    equal[y_true != y_pred] = 1
    counter = equal.sum(axis=0)
    if normalize:
        return counter/y_true.shape[0]
    return counter
    #y_true = np.argsort(y_pred)
    # equal = np.zeros(y_true.shape[0])
    # equal[y_pred == y_true] = 1
    # sumi = equal.sum(axis = 0)
    # if normalize:
    #     return sumi/(y_true.shape[0])
    # else:
    #     return sumi


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Accuracy of given predictions
    """
    equal = np.zeros(y_true.shape[0])
    equal[y_pred == y_true] = 1
    return np.sum(equal)/(y_true.shape[0])


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the cross entropy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Cross entropy of given predictions
    """
    eps = 1e-5
    y_prob = np.clip(y_pred, eps, 1 - eps)
    if y_prob.shape[1] == 1:
        y_prob = np.append(1 - y_prob, y_prob, axis=1)

    if y_true.shape[1] == 1:
        y_true = np.append(1 - y_true, y_true, axis=1)

    return -np.sum(y_true*np.log(y_prob)) / y_prob.shape[0]


def softmax(X: np.ndarray) -> np.ndarray:
    """
    Compute the Softmax function for each sample in given data

    Parameters:
    -----------
    X: ndarray of shape (n_samples, n_features)

    Returns:
    --------
    output: ndarray of shape (n_samples, n_features)
        Softmax(x) for every sample x in given data X
    """
    e_X = np.exp(X)
    sum_exi = np.sum(e_X, axis=0)

    return np.apply_along_axis(lambda x: x/sum_exi, axis=0, arr=X)