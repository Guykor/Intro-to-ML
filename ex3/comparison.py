from models import *
from matplotlib import pyplot as plt
import pandas as pd
import timeit

def draw(m):
    """
    generate sample of size m from the 2-variable normal distribution, and attach
    labels by a specific labeling function
    :param m:  sample size required
    :return: train matrix X (two features) with m samples, label vector y for each sample.
    """
    X = np.random.multivariate_normal(np.zeros(2, ), np.eye(2), m)
    y = np.sign(X @ np.array([0.3, -0.5]) + 0.1)
    return X, y


def draw_points(m):
    """
    given an integer m returns a pair X,y where X is 2xm matrix where eah column
    represents an i.i.d sample from the distribution N(0,I2) and a label vector y,
    by a specified label function.
    make sures that there are at least two samples that are mapped for each label.
    :param m: train set required size
    :return: train data, train_labels
    """
    X, y = draw(m)
    while np.sum(y == 1) <= 1 or np.sum(y == 1) >= m - 1:
        X, y = draw(m)
    return X.T, y


def get_hyp(linspace, intercept, coef_vec):
    """
    given a numpy linspace, intercept(double) and coefficient vector, this function
    computes the hyperplane formula and returns the array of values
    :return: array of values corresponding to the hyper plane y values over the x axis
    (defined by the linspace).
    """
    a = -coef_vec[0] / coef_vec[1]
    return a * linspace - (intercept / coef_vec[1])


def compare_hyperplanes():
    """
    compares hyper plane classifiers over increasing sample sizes, versus the "true"
    hyper plane that labels the syntactic generated data.
    :return: plots a grpah, returns None.
    """
    w = np.array([0.3, -0.5])

    perceptron = Perceptron()
    svm = SVM()

    for m in [5, 10, 15, 25, 70]:
        X, y = draw_points(m)
        xx = np.linspace(np.min(X), np.max(X))
        plt.scatter(X.T[y == 1].T[0, :], X.T[y == 1].T[1, :], color='blue')
        plt.scatter(X.T[y == -1].T[0, :], X.T[y == -1].T[1, :], color='orange')

        plt.plot(xx, get_hyp(xx, 0.1, w), label="True_HP")

        perceptron.fit(X, y)
        plt.plot(xx, get_hyp(xx, perceptron.model[0], perceptron.model[1:]),
                 label="Perceptron HP")
        svm.fit(X, y)
        plt.plot(xx, get_hyp(xx, svm.model.intercept_[0], svm.model.coef_[0]),
                 label="SVM HP")
        plt.legend()
        plt.title("Model HyperPlane comparison for data X~$\mathcal{N}(0,"
                   f"I_{2})$." + f"sample size={m}")
        plt.xlabel("First feature")
        plt.ylabel("Second feature")
        plt.show()


def compare_accuracy():
    """
    this function compares accuracies of three classifiers given a randomized train set
    of increasing size, the metric calculated is each classifier mean accuracy
    according to the train set, as a function of the train sample size.
    the function plots a grpah comparing preformances.
    :return: None
    """
    classifiers = {"perceptron": Perceptron(), "SVM": SVM(), "LDA": LDA()}
    sizes = [5, 10, 15, 25, 70]
    df = pd.DataFrame(0, index=sizes, columns={"perceptron", "SVM", "LDA"},
                      dtype=float)
    for m in sizes:
        for i in range(500):
            X_train, y_train = draw_points(m)
            X_test, y_test = draw_points(10000)
            for col in df.columns:
                classifiers[col].fit(X_train, y_train)
                df.loc[m][col] += classifiers[col].score(X_test, y_test)['accuracy']

    df = np.divide(df, 500)
    plot(df)


def plot(df):
    """
    used to plot the comparison between models accuracy, as a function of sample size.
    :param df: row indices represents sample size, and column the model used for the
    observation.
    :return: None, plots a graph.
    """
    for col in df.columns:
        plt.plot(df.axes[0].tolist(), df[col], label=col)
    plt.legend()
    plt.title("Accuracy rate by classifier over X~$\mathcal{N}(0,I_{2}$)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


def main():
    compare_hyperplanes()
    compare_accuracy()


if __name__ == "__main__":
    main()
