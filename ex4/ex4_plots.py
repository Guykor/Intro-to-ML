"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Plots generation of ex4.

Author: Guy Kornblit, 308224948

"""
from ex4_tools import *
from adaboost import *
from matplotlib import pyplot as plt

T_TRAINING = 500
TRAIN_SIZE = 5000
TEST_SIZE = 200
T_EXPLORE = [5, 10, 50, 100, 200, 500]
NOISES = [0, 0.01, 0.4]


def generate_plots(noise):
    ######################### Q10 ##########################
    """
    plot all the graphs for the exercise regarding adaboost on decision stamp, given a
    noise level for the data genreation (5000  train,200 test)
    """

    X_train, y_train = generate_data(TRAIN_SIZE, noise)
    X_test, y_test = generate_data(TEST_SIZE, noise)
    model = AdaBoost(DecisionStump, T_TRAINING)
    D_T = model.train(X_train, y_train)

    train_error = np.zeros(T_TRAINING)
    test_error = np.zeros(T_TRAINING)
    # build vectors of error given T value.
    for t in np.arange(1, T_TRAINING + 1):
        train_error[t - 1] = model.error(X_train, y_train, t)
        test_error[t - 1] = model.error(X_test, y_test, t)
    #
    plot_Q10(noise, test_error, train_error)
    # plot_Q11(model, noise, X_test, y_test)
    # plot_Q12(model, noise, test_error, X_train, y_train)
    # plot_Q13(D_T, model, noise, X_train, y_train)


def plot_Q13(D_T, model, noise, X_train, y_train):
    D_T = D_T / np.max(D_T) * 10
    plt.plot()
    decision_boundaries(model, X_train, y_train, num_classifiers=T_TRAINING, weights=D_T)
    plt.title(f"train data with size proportional to the final boosted distribution "
              f"\nin adaboost decision stamp classifier (data noise={noise}) ")
    plt.show()


def plot_Q12(model, noise, test_error, X_train, y_train):
    T_hat = np.argmin(test_error) + 1
    decision_boundaries(model, X_train, y_train, num_classifiers=T_hat)
    plt.title("Decision boundary and train data for the best minimizer of the test "
              f"error \ncommittee size={T_hat}, test_error={np.min(test_error)}, "
              f"data noise={noise}")
    plt.show()


def plot_Q11(model, noise, X, y):
    i = 1
    fig = plt.figure()
    for t in T_EXPLORE:
        plt.subplot(2, 3, i)
        decision_boundaries(model, X, y, num_classifiers=t)
        plt.title(f"\n\n\ncommittee size = {t}")
        i += 1
    plt.suptitle("Decision boundaries and test data for Adaboosted Decision stamp "
                 f"classifier \nas a function of it's committee size (noise = {noise})")
    plt.show()


def plot_Q10(noise, test_error, train_error):
    plt.plot(np.arange(1, T_TRAINING + 1), train_error, label="train")
    plt.plot(np.arange(1, T_TRAINING + 1), test_error, label="test")
    plt.xlabel("Committee size (T)")
    plt.ylabel("Misclassification Error")
    plt.title(f"Error rate of Adaboosted Decision Stamp classifier\nas a function of "
              f"the committee size (data noise lv.={noise})")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    for n in NOISES:
        generate_plots(n)
