import numpy as np
import pandas as pd
from pandas import DataFrame
from plotnine import *


def add_noise(data, mu, sigma):
    """
    :param data: numpy array to add gaussian noise to
    :param mu: The expectation of the gaussian noise
    :param sigma: The standard deviation of the gaussian noise
    :return: np array of data with noise
    """
    return data + np.random.normal(loc=mu, scale=sigma)


def fit_model(x, y):
    """
    :param x:  numpy array of dataset samples
    :param y: numpy array of response vector
    :return: A fitted Linear Regression model using sklearn
    """
    model = linear_model.LinearRegression()
    model.fit(x, y_noisy)
    return model


def create_df(x, y, mu, sigma):
    """
    Return a DataFrame with the following columns (exact order and names): x, y, y_noisy, y_hat, r_squared, sigma
        1) y_noisy - should be the y values after noise was added
        2) y_hat - the model's prediction ofr the y values for the given x values.
        3) r_squared - the goodness of fit measurement of the fitted model
        4) sigma - the sigma value the values of this DataFrame were created with
    Hint: On what y values should the model be trained? (In the real world do we ever observe y itself?)
    :param x: The explanatory variable
    :param y: The response variable
    :param mu:  The expectation of the gaussian noise to add
    :param sigma: The standard deviation of the gaussian noise to add
    :return: The created DataFrame
    """
    x = x.reshape(-1, 1)
    y_noisy = add_noise(y, mu, sigma)
    model = fit_model(x, y_noisy)
    y_hat = model.predict(x)

    return pd.DataFrame({'x': x, 'y': y, 'y_noisy': y_noisy, 'y_hat': y_hat,
                         'r_sqared': model.score(x, y_hat), 'sigma': sigma})


def plot_lm_fitting_for_different_noises(x, y, mu, sigmas):
    """"
    :param x: The explanatory variable
    :param y: The response variable
    :param mu:  The expectation of the gaussian noise to add
    :param sigmas: A list of standard deviation noises to apply to response data
    :return: A ggplot with the following:
        1) In black the original x-y points
        2) In red the x-y_noisy points (shape is `*`)
        3) In blue the x-y_hat points (shape is `x`)
        4) A dashed line for the x-y_hat - see geom_line (and linetype='dashed')
        5) Text showing the r_squared
        6) One plot for each value of sigma
    Hint 1: `geom_text` also receices an `aes(...)` argument.
    Hint 2: Recall `facet_wrap` from previous lab
    """
    return pd.concat([create_df(x, y, mu, sigma) for sigma in sigmas], axis=0)


def main():
    x = np.linspace(-2, 100, 20)
    y = 2*x + 1
    # df = create_df(x, y, 1, 100)

    plot_lm_fitting_for_different_noises(x, y, 10, np.arange(1, 27, 5))
