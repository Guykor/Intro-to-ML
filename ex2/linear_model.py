import numpy as np
import pandas as pd
from pyzipcode import ZipCodeDatabase
from plotnine import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# A "json" config area if it was a proper pipeline
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
            'floors', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement',
            'sqft_living15', 'sqft_lot15', 'house_age', 'is_renovated']
categorical_features = ['Auburn', 'Bellevue', 'Black Diamond', 'Bothell', 'Carnation',
                        'Duvall',
                        'Enumclaw', 'Fall City', 'Federal Way', 'Issaquah', 'Kenmore',
                        'Kent',
                        'Kirkland', 'Maple Valley', 'Medina', 'Mercer Island',
                        'North Bend',
                        'Redmond', 'Renton', 'Sammamish', 'Seattle', 'Snoqualmie',
                        'Vashon',
                        'Woodinville', 'Woodland']


def fit_linear_regression(X, y):
    """
    fits a regression model for  design matrix X, and a response vector y.
    :param X: design matrix of dim PxN
    :param y: response vector of dim N
    :return: (coefficient vector, X singular values) both of type np.array
    """
    return np.dot(np.linalg.pinv(X), y), np.linalg.svd(X.drop('intercept', axis=1),
                                                       compute_uv=False)


def predict(X, w):
    """
    given a design matrix and coefficient vector, this function outputs an estimator
    for the response vector ("price").
    :param X: p rows and m column np array
    :param w: coefficient vector of dim px1.
    :return: vector y for the predicted value of each sample (column in X)
    """
    return np.dot(X, w)


def mse(y, y_hat):
    """
    recieves a response vector and prediction vector (numpy arrays) and returns the MSE over the sample.
    :param y: response
    :param y_hat: predicted
    :return: MSE (double)
    """
    return mean_squared_error(y, y_hat)


def load_data(path):
    """
    given a path for the kc_house_data, this function preprocess the data,
    in the matter of cleaning and feature creation, and returns the design matrix and the response matrix
    : param path: path to the kc_house_csv
    :return: design matrix (df), response vector (df)
    """
    data = pd.read_csv(path)
    data = data.dropna(how='any').drop_duplicates()

    # id col
    id_col = data.id.astype(str).apply(lambda x: x.split(r".")[0])
    data.id = id_col
    data = data[~(data.id == '0')]

    # dates
    data['house_age'] = pd.to_datetime(data.date).dt.year - data.yr_built
    data = data[data.house_age >= 0]
    data["is_renovated"] = (data.yr_renovated != 0).astype(int)

    # location
    parser = zip_code_city_parser()
    city = data.zipcode.astype(int).apply(lambda x: parser(x))
    data = pd.concat([data, pd.get_dummies(city)], axis=1)

    # clean garbage
    data = data.query("price > 0 & bedrooms > 0 & sqft_living > 0 & "
                      "sqft_lot > 0 & floors > 0 & sqft_above >= 0 & "
                      "sqft_basement >= 0 & sqft_living15 > 0 & "
                      "sqft_lot15 > 0 & house_age >= 0")

    response = data[['price']]
    data = data.drop(
        columns=["id", "date", "lat", "long", 'yr_renovated', 'yr_built', 'waterfront',
                 "zipcode", "price"])
    data.insert(1, column='intercept', value=np.full((data.shape[0], 1), 1))

    return data, response


def zip_code_city_parser():
    """
    this function used to parse zip code data from the data set,
    returns a zipcode parser function who familier with zipcode relevant for the sample.
    """
    zcdb = ZipCodeDatabase()
    addition = {98077: "Woodland"}

    def parse_zipcode(z):
        try:
            p = zcdb[z]
            return p.city
        except IndexError:
            p = addition[z]
            return p if p else z

    return parse_zipcode


def plot_singular_values(s):
    """
    plot a scree plot of the singular values (ordered) given in s
    :param s: collection of singular values
    :return: None
    """
    df = pd.DataFrame({"values": s}).sort_values(by='values', ascending=False)
    print(ggplot(df, aes(x=range(df.shape[0]), y="np.log(values)")) + \
          geom_point(color='red') + geom_line(color='blue') + \
          labs(x='Index ($\\sigma_{i}$)', y="Singular values (log)",
               title="Singular values (log) Scree plot"))


def test_cumulatve_train(X, y):
    """
    this function splits the data to train (0.75) and test (0.25),
    fits 100 models corresponding to percentage of the train subset,
    and predict values for the syntatic test set and check it's MSE.
    :param X: design matrix of dim (PxM)
    :param y: response vector of dim (Mx1)
    :return: df contatinng percentage of train subset used in model, and mse column for each model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    res = []
    for p in np.arange(1, 101):
        slice_idx = int(X_train.shape[0] * (p / 100))
        w_hat, _ = fit_linear_regression(X_train.iloc[:slice_idx], y_train[:slice_idx])
        y_hat = predict(X_test, w_hat)
        res.append({"percentage": p, "MSE": mse(y_test, y_hat)})
    df = pd.DataFrame(res)
    return df


def plot_mse(df):
    """
    plots a comulative mse for a model by the train set percentage used (cumulative).
    :param df: df containing percentage column and MSE column
    :return: None.
    """
    print(ggplot(df, aes(x='percentage', y='np.divide(MSE, 1000000000)')) \
          + geom_point(color='red') + geom_line(color='blue') \
          + labs(x="Train percentage", y="MSE (in billions)") \
          + ggtitle(f"MSE by train sample size, converges to "
                    f"{(df.iloc[99].MSE / 1000000000).round()} billion"))


def feature_evaluation(X, y):
    """
    this function plots for each feature in the design matrix X
    it's value by their corresponding response (y) value and gives each feature a correlation score.
    :param: design matrix
    :param: response vector.
    :returns prints plots, returns None
    """
    y_std = y['price'].std()
    for column in features:
        df = pd.concat([X[column], y], axis=1)
        corr = (df.cov().iloc[0, 1] / (df[column].std() * y_std)).round(4)
        print(
            ggplot(df) + geom_point(mapping=aes(x=column, y='np.divide(price, 1000000)'),
                                    size=0.01) + \
            ggtitle(
                f"{column} feature values against response\n pearson correlation: {corr}") \
            + labs(x=f'{column}', y='Price (in millions)'))


def main():
    X, y = load_data(r"kc_house_data.csv")
    w_hat, X_singular_vals = fit_linear_regression(X, y)
    plot_singular_values(X_singular_vals)
    plot_mse(test_cumulatve_train(X, y))
    feature_evaluation(X, y)


if __name__ == "__main__":
    main()
