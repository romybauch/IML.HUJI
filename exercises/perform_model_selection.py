from __future__ import annotations

import random

import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) +
    # eps for eps Gaussian noise
    # and split into training- and testing portions
    x = np.linspace(-1.2,2, n_samples)
    y = (x+3)*(x+2)*(x+1)*(x-1)*(x-2)
    y_noise = y + np.random.normal(0, noise, n_samples)

    x_train, y_train, x_test, y_test = split_train_test(pd.DataFrame(x),
                                                    pd.Series(y_noise), 2/3)

    x_train, y_train, x_test, y_test = np.array(x_train).flatten(), \
                                       np.array(y_train), \
                                       np.array(x_test).flatten(), \
                                       np.array(y_test)

    # plot graph
    plot1 = go.Figure([go.Scatter(y=y_train, x=x_train
                      , mode='markers', name="$y_train$"),
                       go.Scatter(y=y_test, x=x_test,
                                  mode='markers', name="$y test$"),
                       go.Scatter(y=y, x=x,
                                  mode='markers + lines', name="$y$")],
              layout=go.Layout(title=r"$\text{data visaluation}$",
                               yaxis_title="y's",
                               xaxis_title="x"))
    plot1.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_error = np.zeros(11)
    val_error = np.zeros(11)

    for i in range(11):
        poli_i = PolynomialFitting(i)
        train_error[i],val_error[i] = cross_validate(poli_i, x_train, y_train,
                                                     mean_square_error, 5)

    plot2 = go.Figure([go.Scatter(y=train_error, x=np.arange(11)
                      , mode='markers+lines', name="$training error$"),
                       go.Scatter(y=val_error, x=np.arange(11),
                                  mode='markers+lines', name="$validation error$"),],
              layout=go.Layout(title=r"5-fold cross validation for polinomial "
                                     r"degrees k=0,..,10",
                               yaxis_title="errors",
                               xaxis_title="polinomial degree"))
    plot2.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and
    # report test error
    best_k = np.argmin(val_error)
    train_k_poli = PolynomialFitting(int(best_k)).fit(x_train,y_train)
    test_err = round(mean_square_error(train_k_poli.predict(x_test),y_test),2)

    print("number of samples: " + str(n_samples) + ", noise: " + str(noise))
    print("value of k* = " + str(int(best_k)))
    print("test error of the fitted model = " + str(test_err))


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best
    fitting regularization parameter values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the
        algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing
    # portions
    X, y = datasets.load_diabetes(return_X_y=True)
    x_train, x_test, y_train, y_test = X[:50], X[50:], y[:50], y[50:]

    # Question 7 - Perform CV for different values of the regularization
    # parameter for Ridge and Lasso regressions
    lambdas = np.linspace(0,2,n_evaluations)
    train_err_ridge = np.zeros(n_evaluations)
    val_err_ridge = np.zeros(n_evaluations)
    train_err_lasso = np.zeros(n_evaluations)
    val_err_lasso = np.zeros(n_evaluations)

    for i,lam in enumerate(lambdas):
        ridge_lam = RidgeRegression(lam)
        lasso_lam = Lasso(lam)

        train_err_ridge[i],val_err_ridge[i] = cross_validate(ridge_lam,
                                                             x_train, y_train,
                                            mean_square_error, 5)
        train_err_lasso[i],val_err_lasso[i] = cross_validate(lasso_lam,
                                                             x_train, y_train,
                                            mean_square_error, 5)

    plot_lasso = go.Figure([go.Scatter(y=train_err_lasso, x=lambdas
                      , mode='markers+lines', name="$training error$"),
                       go.Scatter(y=val_err_lasso, x=lambdas,
                                  mode='markers+lines', name="$lasso errors$"),],
              layout=go.Layout(title=r"lasso err",
                               yaxis_title="errors",
                               xaxis_title="lamda"))
    plot_lasso.show()

    plot_ridge = go.Figure([go.Scatter(y=train_err_ridge, x=lambdas
                      , mode='markers+lines', name="$training error$"),
                       go.Scatter(y=val_err_ridge, x=lambdas,
                                  mode='markers+lines', name="$ridge errors$"),],
              layout=go.Layout(title=r"ridge err",
                               yaxis_title="errors",
                               xaxis_title="lamda"))
    plot_ridge.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lambda_ridge = lambdas[np.argmin(val_err_ridge)]
    best_lambda_lasso = lambdas[np.argmin(val_err_lasso)]

    print("ridge: lambda that achieved the best validation error- " +
          str(round(best_lambda_ridge, 3)))
    print("lasso: lambda that achieved the best validation error- " +
          str(round(best_lambda_lasso, 3)))

    trained_ridge = RidgeRegression(best_lambda_ridge).fit(x_train, y_train)
    trained_lasso = Lasso(best_lambda_lasso).fit(x_train, y_train)
    trained_linear = LinearRegression().fit(x_train, y_train)

    ridge_err = mean_square_error(trained_ridge.predict(x_test),y_test)
    lasso_err = mean_square_error(trained_lasso.predict(x_test), y_test)
    linear_err = mean_square_error(trained_linear.predict(x_test),y_test)

    print("ridge test error: " + str(ridge_err))
    print("lasso test error: " + str(lasso_err))
    print("linear test error: " + str(linear_err))


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(100,0)
    select_polynomial_degree(1500,10)
    select_regularization_parameter()

