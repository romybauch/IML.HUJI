import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange,
                                       density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1],
                                 mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None],
                                              List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's
    value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding
        the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values, weights = [], []

    def call_back(**kwargs):
        values.append(kwargs["val"])
        weights.append(kwargs["weights"])
        return

    return call_back, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for et in etas:
        # l1
        l1_et = L1(init.copy())
        lr_et1 = FixedLR(et)
        call_et1, val_et1, weights_et1 = get_gd_state_recorder_callback()
        gr_et1 = GradientDescent(learning_rate= lr_et1,callback= call_et1,
                                 out_type="best")
        gr_et1.fit(l1_et, None, None)
        plot_et = plot_descent_path(L1, np.array(weights_et1),
                                    "l1 with eta: "+str(et))
        plot_et.show()
        # convergence rate
        plot_convergence1 = go.Figure([go.Scatter(y=val_et1,
                          x=np.arange(len(val_et1)) , mode='markers'),],
              layout=go.Layout(title=r"convergence rate of L1 with etas: "+str(et),
                               yaxis_title="convergence value",
                               xaxis_title="iteration number"))
        plot_convergence1.show()

        # l2
        l2_et = L2(init.copy())
        lr_et2 = FixedLR(et)
        call_et2, val_et2, weights_et2 = get_gd_state_recorder_callback()
        gr_et2 = GradientDescent(learning_rate=lr_et2, callback=call_et2,
                                 out_type="best")
        gr_et2.fit(l2_et, None, None)
        plot_et = plot_descent_path(L2, np.array(weights_et2),
                                    "l2 with eta: " + str(et))
        plot_et.show()
        plot_convergence2 = go.Figure([go.Scatter(y=val_et2,
                          x=np.arange(len(val_et2)), mode='markers'),],
              layout=go.Layout(title=r"convergence rate of L2 with etas: "+str(et),
                               yaxis_title="convergence value",
                               xaxis_title="iteration number"))
        plot_convergence2.show()


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the
    # exponentially decaying learning rate
    all_rate = []
    all_weight1 = []
    all_weight2 = []
    for gm in gammas:
        # l1
        l1 = L1(init.copy())
        lr1 = ExponentialLR(eta, gm)
        callback1, val_et1, weights1 = get_gd_state_recorder_callback()
        gr_et1 = GradientDescent(learning_rate=lr1,
                                 callback=callback1,
                                 out_type="best")
        gr_et1.fit(l1, None, None)

        all_rate.append(val_et1)

        if gm==0.95:
            all_weight1.append(weights1)

            l2 = L2(init.copy())
            lr2 = ExponentialLR(eta, gm)
            callback2, val_et2, weights2 = get_gd_state_recorder_callback()
            gr_et2 = GradientDescent(learning_rate=lr2,
                                     callback=callback2,
                                     out_type="best")
            gr_et2.fit(l2, None, None)
            all_weight2.append(weights2)

    plot_convergence = go.Figure([go.Scatter(y=all_rate[0],
                                      x=np.arange(len(all_rate[0])),
                                      mode="markers+lines", marker_color = 'red'),
                                  go.Scatter(y=all_rate[1],
                                     x=np.arange(len(all_rate[1])),
                                     mode="markers+lines", marker_color = "blue"),
                                  go.Scatter(y=all_rate[2],
                                     x=np.arange(len(all_rate[2])),
                                     mode="markers+lines", marker_color = "green"),
                                  go.Scatter(y=all_rate[3],
                                     x=np.arange(len(all_rate[3])),
                                     mode="markers+lines", marker_color = "pink"),
                                  ],
                          layout=go.Layout(
                              title=r"convergence rate of L1 with different "
                                    r"gammas",
                              yaxis_title="convergence value",
                              xaxis_title="iteration number"))

    # Plot algorithm's convergence for the different values of gamma
    plot_convergence.show()

    # Plot descent path for gamma=0.95
    plot_descent_path(L1, np.array(all_weight1[0]),
                      "gd for gemma = 0.95 with L1").show()
    plot_descent_path(L2, np.array(all_weight2[0]),
                      "gd for gemma = 0.95 with L2").show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train-
    and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
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
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd,
                            train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    from sklearn.metrics import roc_curve, auc
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), \
                                       np.array(X_test), np.array(y_test)

    # Plotting convergence rate of logistic regression over SA heart disease
    # data
    logistic_reg = LogisticRegression(include_intercept=True,
                                      solver=GradientDescent(
                                          learning_rate=FixedLR(1e-4),
                                          max_iter=20000)).fit(X_train, y_train)
    predict_prob = logistic_reg.predict_proba(X_train)

    fpr, tpr, thresholds = roc_curve(y_train, predict_prob)

    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                         line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds,
                         name="", showlegend=False, marker_size=5,

                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: "
                                       "%{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(
            title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
            xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
            yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()

    temp = np.argmax(tpr-fpr)
    best_a = round(thresholds[np.argmax(tpr-fpr)],5)
    print("best a: " + str(best_a))

    logistic_reg.alpha_ = best_a
    loss_best_a = logistic_reg.loss(X_test, y_test)
    print("loss by best a: " + str(loss_best_a))

    # Fitting l1- and l2-regularized logistic regression models, using
    # cross-validation to specify values of regularization parameter
    from IMLearn.model_selection import cross_validate
    from IMLearn.metrics import loss_functions
    logi_l1 = LogisticRegression(solver=GradientDescent(
                                           learning_rate=FixedLR(1e-4),
                                           max_iter=20000),penalty="l1", alpha=0.5)

    lamdot = [0.001,0.002,0.005,0.01,0.02,0.05,0.1]
    val_err1 = []

    for lam in lamdot:
        logi_l1.lam_ = lam
        train_err, val_err = cross_validate(logi_l1, X_train, y_train,
                                    loss_functions.misclassification_error)
        val_err1.append(val_err)

    best_lam = lamdot[np.argmin(val_err1)]
    logi_l1.lam_ = best_lam
    print("best lambda for L1 is: " + str(best_lam))
    print("test error: " + str(logi_l1.loss(X_test, y_test)))



if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
