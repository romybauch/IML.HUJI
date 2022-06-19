import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics.loss_functions import accuracy


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape 
    (num_samples). num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    # Question 1: Train- and test errors of AdaBoost in noiseless case
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), \
                                           generate_data(test_size, noise)

    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)

    error_test = np.zeros(n_learners)
    error_train = np.zeros(n_learners)

    for i in range(1,n_learners+1):
        error_test[i-1] = adaboost.partial_loss(test_X, test_y, i)
        error_train[i-1] = adaboost.partial_loss(train_X, train_y, i)

    x = list(range(1, n_learners + 1))
    go.Figure([go.Scatter(x=x, y=error_train, mode='markers+lines',
                          name=r'$\widehat\mu$'),
               go.Scatter(x=x, y=error_test, mode='lines', name=r'$\mu$')],
              layout=go.Layout(
                  title=r"$\text{The train and test error as a function of "
                        r"the number of fitted learners}$",
                  xaxis_title="$\\text{number of iterations}$",
                  yaxis_title="MSE",
                  height=500)).show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + \
           np.array([-.1, .1])
    symbols = np.array(['x', 'circle', 'diamonds'])
    fig2 = make_subplots(rows=2, cols=2, subplot_titles=[f"Decision boundary"
                                             f"for ensemble of size {m}"
                                             for i,m in enumerate(T)],
                         horizontal_spacing= 0.2, vertical_spacing= 0.1)

    for i, n in enumerate(T):
        fig2.add_traces([go.Scatter(
            x=test_X[:,0], y=test_X[:,1],
            mode="markers", showlegend=False,
            marker=dict(color=test_y, symbol='diamond',
                        line=dict(color="black",width=1)),
        ),
            decision_surface(lambda x: adaboost.partial_predict(x,n),
                                lims[0], lims[1], showscale=False)],
            rows=(i // 2) + 1, cols=(i % 2) + 1
        )

    fig2.update_layout(title=f"Decision boundary for different ensemble size",
                       margin=dict(t=100), width=1200, height=1000)

    fig2.show()

    # Question 3: Decision surface of best performing ensemble
    min_err_it = int(np.argmin(error_test))
    y_pred = adaboost.partial_predict(test_X, min_err_it)
    acc = accuracy(test_y, y_pred)
    symbol_decide = np.where(test_y < 0, 0, 1)
    fig3 = go.Figure().add_traces(
        [
            decision_surface(lambda d: adaboost.partial_predict(d, min_err_it),
                             lims[0], lims[1], showscale=False),
            go.Scatter(x=test_X[:,0], y=test_X[:,1], mode="markers",
                       showlegend=False,
                       marker=dict(color=test_y, symbol=symbols[symbol_decide],
                                   colorscale=[custom[0], custom[-1]],
                                   line=dict(color="pink", width=1)))
        ])
    fig3.update_layout(title="Decision surface with " + str(min_err_it)+
                             " ensemble reach lowest test error with accuracy "
                             + str(acc),
                       yaxis1_range=[-1,1], xaxis1_range=[-1,1],
                       margin=dict(t=100), width=1200, height=1000)
    fig3.show()

    # Question 4: Decision surface with weighted samples
    y_train_plot = np.where(train_y < 0, 0, 1)
    D_plot = adaboost.D_ / np.max(adaboost.D_) * 5
    fig4 = go.Figure().add_traces(
        [
            decision_surface(adaboost.predict,
                             lims[0], lims[1], showscale=False),
            go.Scatter(x=train_X[:,0], y=train_X[:,1], mode="markers",
                       showlegend=False,
                       marker=dict(color=train_y, symbol=symbols[y_train_plot],
                                   colorscale=[custom[0], custom[-1]],
                                   size=D_plot,
                                   line=dict(color=train_y, width=1)))
        ])
    fig4.update_layout(title="Training set with point size and color according"
                             " to weights and labels",
                       yaxis1_range=[-1,1], xaxis1_range=[-1,1],
                       margin=dict(t=100), width=1200, height=1000)
    fig4.show()



if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)