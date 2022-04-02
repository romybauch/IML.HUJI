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
    x = pd.DataFrame(pd.read_csv(filename, usecols=np.arange(2,17)))

    x = x.where(x>=0, other=pd.NaT).dropna()
    x = pd.get_dummies(x,columns=['zipcode'])

    #x['yr_built'] = 2022-x['yr_built']
    x.loc[x['yr_renovated'] > 0, 'yr_since_renovated'] = x['yr_renovated']
    x.loc[x['yr_renovated'] <= 0, 'yr_since_renovated'] = x['yr_built']
    x = x.where(x['yr_since_renovated']>0, other=pd.NaT).dropna()
    #x = x.where(x[['yr_renovated','bedrooms','bathrooms','sqrf_living','sqft_lot']]>0, other=pd.NaT).dropna()


    #pd.DataFrame(x[['yr_since_renovated',"yr_renovated"]]).to_csv("./temp_labels.csv")

    del x['yr_built']
    del x['yr_renovated']
    x = x.astype(float)
    y= x['price']
    del x['price']
    return x,y


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
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
    # todo: insert y to x
    X.insert(0,'price', y)
    cov_XY = X.cov()['price']
    std_y = y.std()
    X.sort_values(by=['price'])
    #todo: change to readable before hagash
    # for col in X.columns:
    #     corr = float(cov_XY[col]/(std_y*(X[col].std())))

        # figi = go.Figure([go.Scatter(x=X['price'], y= X[col], mode='markers')],
        #           layout=go.Layout(title=f"feature is: {col} and the pearson "
        #                                  f"correlation between is: {corr}"
        #                            , xaxis_title = "price"
        #                            , yaxis_title = f"number of {col}", width= 1000))
        # pio.write_image(figi, output_path+'/'+ col + '.png')

    # corr = float(cov_XY['yr_since_renovated']/(std_y*(X['yr_since_renovated'].std())))
    #
    # figi = go.Figure([go.Scatter(x=X['price'], y= X[col], mode='markers')],
    #           layout=go.Layout(title=f"feature is: {col} and the pearson "
    #                                  f"correlation between is: {corr}"
    #                            , xaxis_title = "price"
    #                            , yaxis_title = f"number of {col}", width= 1000)).show()
    # pio.write_image(figi, output_path+'/'+ col + '.png')

    del X['price']

if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of housing prices dataset
    X, y_true = load_data(r"C:\Users\romyb\Documents\iml\IML.HUJI\datasets\house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y_true, r"C:\Users\romyb\Documents\iml\IML.HUJI\exercises\house_price_predic_plot2")

    # Question 3 - Split samples into training- and testing sets.
    print("x shape:")
    print(X.shape)
    train_x, train_y, test_x, test_y = split_train_test(X,y_true)

    # print("train x: ")
    # print(train_x.shape)
    # print("train y: ")
    # print(train_y.shape)
    #
    # print("test x: ")
    # print(test_x.shape)
    # print("test y: ")
    # print(test_y.shape)


    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    to_train = X
    to_train.insert(0,'price', y_true)
    for i in range(10,101):
        for j in range(10):
            temp_x = to_train.sample(n=i)
            temp_y = temp_x['price']
            del temp_x['price']
            lini_temp = LinearRegression(False)
            lini_temp._fit(temp_x, temp_y)





