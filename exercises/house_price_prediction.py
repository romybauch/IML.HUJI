from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> (pd.DataFrame, pd.Series):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    x = pd.DataFrame(pd.read_csv(filename, usecols=np.arange(2,17))
                     .drop_duplicates())
    x = x.where(x>=0, other=pd.NaT).dropna()
    x = x[x['zipcode'] > 0]
    x = pd.get_dummies(x, columns=['zipcode'])
    x.loc[x['yr_renovated'] > 0, 'yr_since_renovated'] = x['yr_renovated']
    x.loc[x['yr_renovated'] <= 0, 'yr_since_renovated'] = x['yr_built']
    del x['yr_built']
    del x['yr_renovated']
    x = x[x['yr_since_renovated'] > 0]
    x = x[x['sqft_living'] > 0]
    x = x[x['sqft_lot'] > 0]
    x = x.astype(float)

    x.to_csv(r"C:\Users\romyb\Documents\iml\IML.HUJI\exercises\temp_labels.csv")
    y = x['price']
    del x['price']
    return x,y


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") \
        -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    X.insert(0,'price', y)
    cov_XY = X.cov()['price']
    std_y = y.std()
    X.sort_values(by=['price'])

    for col in X.columns:
        corr = float(cov_XY[col]/(std_y*(X[col].std())))
        print(col)
        print(corr)
        # figi = go.Figure([go.Scatter(x=X['price'], y= X[col], mode='markers')],
        #           layout=go.Layout(title=f"feature is: {col} and the pearson "
        #                                  f"correlation between is: {corr}"
        #                            , xaxis_title = "price"
        #                            , yaxis_title = f"number of {col}", width= 1000))
        # pio.write_image(figi, output_path+'/'+ col + '.png')
    del X['price']


def fit_test_model(x_train: pd.DataFrame, y_train: pd.Series, x_test, y_test):
    # For every percentage p in 10%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of

    sample_num = 10
    lini_model = LinearRegression(True)
    mean_loss, std_loss = np.zeros(91), np.zeros(91)
    x_test, y_test = np.array(x_test), np.array(y_test)
    a = np.arange(10,100,1)
    for p in a:
        cur_loss = np.zeros(sample_num)
        for j in range(sample_num):
            cur_x = x_train.sample(frac=p/100)
            cur_y = y_train.loc[cur_x.index]
            cur_x, cur_y = np.array(cur_x), np.array(cur_y)
            lini_model.fit(cur_x, cur_y)
            cur_loss[j] = lini_model.loss(x_test, y_test)

        mean_loss[p-10] = np.mean(cur_loss)
        std_loss[p-10] = np.std(cur_loss)

    up_func = mean_loss + (2 * std_loss)
    down_func = mean_loss - (2 * std_loss)
    a = a/100
    fig = go.Figure(
        (go.Scatter(x=a, y=mean_loss, mode="markers+lines",
                    name="Mean Prediction", line=dict(dash="dash"),
                    marker=dict(color="green", opacity=.7),
                    ),
         go.Scatter(x=a, y=up_func, fill=None,
                    mode="lines", line=dict(color="lightgrey"),
                    showlegend=False),
         go.Scatter(x=a, y=down_func, fill='tonexty',
                    mode="lines", line=dict(color="lightgrey"),
                    showlegend=False),),
          layout=go.Layout(title="mean loss as a function of p% and condifence"
                                 " interval"
                           , xaxis_title = "p% of train set"
                           , yaxis_title = "mean loss"))
    fig.show()


if __name__ == '__main__':

    np.random.seed(0)

    # Question 1 - Load and preprocessing of housing prices dataset
    X, y_true = load_data(r"C:\Users\romyb\Documents\iml\IML.HUJI\datasets\house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y_true, r"C:\Users\romyb\Documents\iml\IML.HUJI\exercises\house_price_predic_plot2")

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(X,y_true)

    # Question 4
    fit_test_model(train_x, train_y, test_x, test_y)

