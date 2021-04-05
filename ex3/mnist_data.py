import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import time


def load_data():
    """
    Loads the mnist database, only for  0's and 1's.
    :return: filtered data set, X_train, y_train, X_test, y_test
    """
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    train_images = np.logical_or(y_train == 0, y_train == 1)
    test_images = np.logical_or(y_test == 0, y_test == 1)

    return (X_train[train_images], y_train[train_images],
            X_test[test_images], y_test[test_images])


def plot_images(X_train, y_train):
    """
    function that enables sampled view of three images of each label
    :param X_train: train set
    :param y_train: train labels.
    :return: None
    """
    zeros = X_train[y_train == 0]
    ones = X_train[y_train == 1]

    for i in [0, 1, 2]:
        plt.imshow(zeros[i])
        plt.show()
        plt.imshow(ones[i])
        plt.show()


def rearrange_data(X):
    """
    Given a data as a tensor of size m x 28 x 28 returns a New matrix of size m x 784
    with the same data.
    :param X: tensor of size m x 28 x 28
    :return: matrix of size m x 784
    """
    return np.reshape(X, (X.shape[0], 784))


def draw_data(m, X, y):
    """
    randomly samples from X and y (train set) randomly, make sures it has at least two
    samples of each label.
    :return: X sample of size m, and corresponding label vector of size m.
    """
    indices = np.random.randint(0, X.shape[0], m)
    X_rand, y_rand = X[indices], y[indices]
    while np.sum(y_rand == 1) <= 1 or np.sum(y_rand == 1) >= m - 1:
        indices = np.random.randint(0, X.shape[0], m)
        X_rand, y_rand = X[indices], y[indices]
    return X_rand, y_rand


def compare_classifiers(X_train, y_train, X_test, y_test):
    """
    given the mnist data set, this function compares and plot mean accuracies of
    classifiers.
    """
    classifiers = {"Logistic-regression": LogisticRegression(),
                   "Soft-svm (reg=1)": SVC(),
                   "Decision_tree (max_depth-7)": DecisionTreeClassifier(max_depth=7),
                   "7 Nearest Neighbours": KNeighborsClassifier(n_neighbors=7)}

    sizes = [50, 100, 300, 500]
    df = pd.DataFrame(0, index=sizes,
                      columns={"Logistic-regression", "Soft-svm (reg=1)",
                               "Decision_tree (max_depth-7)",
                               "7 Nearest Neighbours"}, dtype=float)
    elapsed_time = df.copy()
    for m in sizes:
        for col in df.columns:
            for _ in range(50):
                train_set, train_labels = draw_data(m, X_train, y_train)
                start = time.time()
                classifiers[col].fit(train_set, train_labels)
                df.loc[m][col] += classifiers[col].score(X_test, y_test)
                end = time.time()
                elapsed_time.loc[m][col] += (end - start)

    df = np.divide(df, 50)
    plot_comparison(df)

    elapsed_time = np.exp(np.divide(elapsed_time, 50))
    elapsed_time.plot()
    plt.title("Elapsed run time for each model to train and \n"
              "predict as a function of sample size")
    plt.xlabel("Sample size")
    plt.ylabel("Run time (seconds), exp scaled")
    plt.show()


def plot_comparison(df):
    """
    plot the accuracies for each classifier as a function of sample size.
    :param df: data frame containing rows corresponding to sample size, and column as
    the mean accuracies per classifier.
    :return:  None
    """
    for col in df.columns:
        plt.plot(df.axes[0].tolist(), df[col], label=col)
    plt.legend()
    plt.title("Classifiers mean accuracy by size of train sample - mnist problem")
    plt.xlabel("Size of train sample")
    plt.ylabel("Mean accuracy")
    plt.show()


def main():
    """
    this module compares classifiers accuracy over the mniset database, over label 0
    and 1.
    :return: None
    """
    X_train, y_train, X_test, y_test = load_data()
    X_train = rearrange_data(X_train)
    X_test = rearrange_data(X_test)
    compare_classifiers(X_train, y_train, X_test, y_test)
    # run_times.to_excel(r"runtimes.xlsx")


if __name__ == "__main__":
    main()
