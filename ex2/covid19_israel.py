from linear_model import fit_linear_regression, predict
import numpy as np
import pandas as pd
from plotnine import *


def load_data(path):
    """
    this function loads and preproccesses the covid 19 data from israel
    :param path: path to the file
    :return: df
    """
    data = pd.read_csv(path)
    data['log_detected'] = np.log(data.detected)
    data['intercept'] = np.full(data.shape[0], 1)
    return data


def plot_predictions(data):
    """
    this function produces two plots for the exponential growth of the covid19 rate.
    the first plot compares between a linear prediction of number of detected cases
    ageinst log(detected) cases, and the second compares exp(y_hat or prediction
    values) against number of detected cases.
    :return: None
    """

    print(ggplot(data) + geom_point(
        aes(x='day_num', y='log_detected', color=['log_detected'])) + \
          geom_line(aes(x='day_num', y='regression_model', color=['regression_model'])) + \
          labs(title="Covid19 log detected cases per day estimator vs reality", color=''))

    print(ggplot(data) + geom_point(aes(x='day_num', y='detected', color=['detected']))
          + geom_line(aes(x='day_num', y='np.exp(regression_model)',
                          color=['regression_model'])) + \
          labs(title="Covid19 detected cases per day estimator vs reality", color=''))


def main():
    data = load_data(r"covid19_israel.csv")
    w_hat, _ = fit_linear_regression(data[['intercept', 'day_num']],
                                     data[['log_detected']])

    data['regression_model'] = predict(data[['intercept', 'day_num']], w_hat)
    plot_predictions(data)


if __name__ == "__main__":
    main()
