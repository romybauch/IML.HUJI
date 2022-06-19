import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from datetime import datetime, date
import plotly.graph_objects as go

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    x = pd.read_csv(filename, parse_dates=['Date']).dropna().drop_duplicates()

    # sure that the date match - year,month,day and the dates make sence
    x = x[x['Temp']>-70]
    x = x[x['Year'] > 0]
    x = x[x['Year'] < 2023]
    x = x[x['Month'] > 0]
    x = x[x['Month'] < 13]
    x = x[x['Day'] > 0]
    x = x[x['Day'] < 32]

    x = x[x['Day'] == pd.DatetimeIndex(x['Date']).day]
    x = x[x['Month'] == pd.DatetimeIndex(x['Date']).month]
    x = x[x['Year'] == pd.DatetimeIndex(x['Date']).year]

    x['DayOfYear'] = x['Date'].dt.dayofyear

    # make dummies
    x = pd.get_dummies(x, columns=['City'])

    x.to_csv(r"C:\Users\romyb\Documents\iml\IML.HUJI\exercises\try.csv")

    return x

def quest_2(x: pd.DataFrame):
    # fig1
    israel = x[x['Country'] == 'Israel']
    fig1 = px.scatter(israel, x='DayOfYear', y='Temp', color = 'Year',
               title= "relation between Israel and temperature")
    #fig1.show()

    # fig2
    temp_std = israel.groupby('Month')['Temp'].std().rename('std')
    fig2 = px.bar(temp_std, title="standard deviation of each month")
    fig2.update_traces(marker_color='wheat')
    #fig2.show()


def quest_3(x: pd.DataFrame):
    tempi = x.groupby(['Country', 'Month']).agg(
        {'Temp': ['mean', 'std']}).reset_index()
    tempi.columns = ['Country', 'Month', 'mean', 'std']
    fig3 = px.line(tempi, x='Month', y='mean',
                  color=tempi['Country'].astype(str),
                  title="Average temp for month for each country",
                   error_y='std')
    #fig3.show()


def quest_4(x: pd.DataFrame):
    israel = x[x['Country'] == 'Israel']
    y_israel = israel['Temp']
    x_israel = israel['DayOfYear']
    train_x, train_y, test_x, test_y = split_train_test(x_israel, y_israel)

    ks = np.arange(1,11,1)
    k_loss = np.zeros(10)
    for k in ks:
        estimator = PolynomialFitting(k)
        estimator.fit(train_x.to_numpy().flatten(), train_y.to_numpy())
        k_loss[k-1] = round(estimator.loss(test_x.to_numpy().flatten(),
                                     test_y.to_numpy()),2)
        #print("the k is: "+str(k)+", mse is: "+ str(k_loss[k-1]))

    fig4 = px.bar(x=ks,y=k_loss, title="test error recorded for each value of k",
                  labels={"x":"value of k", "y":"MSE"})
    fig4.show()

    return np.amin(k_loss)


def quest_5(x: pd.DataFrame, best_k: float):
    israel_df = x[x['Country'] == 'Israel']
    estimator = PolynomialFitting(int(best_k))
    estimator.fit(israel_df['DayOfYear'].to_numpy().flatten(),
                  israel_df['Temp'].to_numpy())

    africa_df = x[x['Country'] == 'South Africa']
    neth_df = x[x['Country'] == 'The Netherlands']
    jordan_df = x[x['Country'] == 'Jordan']

    countries_loss = np.zeros(3)
    countries_loss[0] = estimator.loss(africa_df['DayOfYear'].to_numpy()
                                       .flatten(),africa_df['Temp'].to_numpy())
    countries_loss[1] = estimator.loss(neth_df['DayOfYear'].to_numpy()
                                       .flatten(),
                                       neth_df['Temp'].to_numpy())
    countries_loss[2] = estimator.loss(jordan_df['DayOfYear'].to_numpy()
                                       .flatten(),
                                       jordan_df['Temp'].to_numpy())

    fig5 = px.bar(x=['South Africa','The Netherlands', 'Jorden'],
                  y=countries_loss,
                  title="Israel model's error over each of the other countries",
                  labels={"x": "country", "y": "MSE"})
    fig5.show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    x = load_data(r"C:\Users\romyb\Documents\iml\IML.HUJI\datasets\City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    quest_2(x)

    # Question 3 - Exploring differences between countries
    quest_3(x)

    # Question 4 - Fitting model for different values of `k`
    best_k = quest_4(x)

    # Question 5 - Evaluating fitted model on different countries
    quest_5(x, best_k)