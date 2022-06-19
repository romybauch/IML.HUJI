import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers.
    File is assumed to be an ndarray of shape (n_samples, 3) where the first 2
    columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the
    linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss
    values (y-axis) as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X,Y = load_dataset(f'../datasets/{f}')
        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def my_call_back(fit: Perceptron, x: np.ndarray, y: int):
            losses.append(fit.loss(X,Y))
        perc_model = Perceptron(callback=my_call_back)
        perc_model.fit(X,Y)
        perc_model.predict(X)
        print(losses)

        # Plot figure of loss as function of fitting iteration
        iters = np.arange(0, len(losses))

        go.Figure([go.Scatter(x=iters, y=np.array(losses),mode='lines')],
              layout=go.Layout(
              title="Perceptrone algorithm loss as a function of it iterations",
              xaxis_title="number of iterations",
              yaxis_title="loss", height=400)).show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else \
        (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines",
                      marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and
    gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X,y = load_dataset(f'../datasets/{f}')

        # Fit models and predict over training set
        lda_model = LDA()
        lda_model.fit(X,y)
        gnb_model = GaussianNaiveBayes()
        gnb_model.fit(X,y)
        # Plot a figure with two suplots, showing the Gaussian Naive Bayes
        # predictions on the left and LDA predictions on the right. Plot title
        # should specify dataset used and subplot titles should specify
        # algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        accur_model = [round(accuracy(y, gnb_model.predict(X)),2),
                       round(accuracy(y, lda_model.predict(X)),2)]

        lims = np.array([X.min(axis=0), X.max(axis=0)]).T + np.array([-.4, .4])
        models = [gnb_model,lda_model]
        model_names = ["Gaussian Naive Bayes","LDA"]
        symbols = np.array(["circle", "star", "diamond", "X"])

        fig = make_subplots(rows=2, cols=3,
                            subplot_titles=[f"{m} <br> Accuracy is: "
                                            f"{accur_model[i]}" for i,m in
                                            enumerate(model_names)],
                            horizontal_spacing=0.01, vertical_spacing=.03)
        for i, m in enumerate(models):
            fig.add_traces([decision_surface(m.predict, lims[0], lims[1],
                                             showscale=False),
                            go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                       showlegend=False,
                                       marker=dict(color=m.predict(X),
                                                   symbol=symbols[y],
                                                   colorscale=[custom[0],
                                                               custom[-1]],
                                                   line=dict(color="black",
                                                             width=1)))],
                           rows=(i // 3) + 1, cols=(i % 3) + 1)

        fig.update_layout(
            title=rf"$\textbf{{Decision Boundaries Of Models - {f} Dataset}}$",
            margin=dict(t=100)) \
            .update_xaxes(visible=False).update_yaxes(visible=False)
        fig.show()

        # Add traces for data-points setting symbols and colors

        # Add `X` dots specifying fitted Gaussians' means
        for i, m in enumerate(models):
            fig.add_traces([go.Scatter(x=m.mu_.T[0], y=m.mu_.T[1],
                                       mode='markers', showlegend=False,
                                       marker=dict(color="green", opacity=.7,
                                                   symbol='x'))],
            rows = (i // 3) + 1, cols = (i % 3) + 1)
        fig.update_layout(
            title=rf"$\textbf{{Decision Boundaries Of Models - {f} Dataset}}$",
            margin=dict(t=100)) \
            .update_xaxes(visible=False).update_yaxes(visible=False)
        fig.show()

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(3):
            fig.add_trace(row=1, col=1, trace=get_ellipse(models[0].mu_[i],
                                              np.diag(models[0].vars_[i])))
            fig.add_trace(row=1, col=2, trace=get_ellipse(models[1].mu_[i],
                                                          models[1].cov_))
        fig.update_layout(
            title=rf"$\textbf{{Decision Boundaries Of Models - {f} Dataset}}$",
            margin=dict(t=100)) \
            .update_xaxes(visible=False).update_yaxes(visible=False)
        fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
