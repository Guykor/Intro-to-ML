import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


class model:
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        """
        Given a training set X of dim (d,m) and label y vector {1,-1}^m,
        this method learns the parameters of the model and stores the trained model
        in self.model.
        :param X: 2 dim array
        :param y: 2 dim vector (m,1)
        :return: None
        """
        pass

    def predict(self, X):
        """
        Given an unlabeled test set X of shape (d,m'), predicts the label of each sample.
        Returns a vector of predicted labels y {1,-1}^m'.
        :param X: 2 dim array
        :return: predicte labels vector, array of dim 2
        """
        pass

    def __calc_metrics(self, test_set, true_labels):
        """

        :param test_set:
        :param true_labels:
        :return:
        """
        positives = (true_labels == 1).sum()
        negatives = true_labels.shape[0] - positives
        y_hat = self.predict(test_set)

        true_positives = np.sum(np.logical_and(y_hat == 1, true_labels == 1))
        false_positives = np.sum(np.logical_and(y_hat == 1, true_labels != 1))
        true_negative = np.sum(np.logical_and(y_hat == -1, true_labels == -1))
        false_negative = np.sum(np.logical_and(y_hat == -1, true_labels != -1))
        return positives, negatives, true_positives, false_positives, true_negative, false_negative

    def score(self, X, y):
        """
        Given an unlabeled test set X of shape (d,m') and true labels vector y {-1,1}^m'
        of this test set, returns a dictionary with the following fields"
        • num samples: number of samples in the test set
        • error: error (misclassification) rate
        • accuracy: accuracy
        • FPR: false positive rate
        • TPR: true positive rate
        • precision: precision
        • recall: recall
        :param X: 2 dim numpy array
        :param y: 2 dim nupmy vector
        :return: dictionary
        """
        if self.model is None:
            print("Train first")
            return
        p, n, tp, fp, tn, fn = self.__calc_metrics(X, y)
        result = dict()
        result['num_samples'] = X.shape[1]
        result['error'] = (fp + fn) / (p + n)
        result['accuracy'] = (tp + tn) / (p + n)
        result['FPR'] = fp / n
        result['TPR'] = tp / p
        result['precision'] = tp / (tp + fp)
        result['recall'] = tp / p

        return result


class Perceptron(model):

    def fit(self, X, y):
        """
        improve coefficient vector iteratively given a train data and it's labels vector.
        updates model variable self.model.
        :param X: train set of dim (d,m)
        :param y: train label of dim (m,)
        :return: None
        """
        train_set = np.insert(X, 0, values=np.full(X.shape[1], 1), axis=0)
        w = np.full((train_set.shape[0],), 0)

        y_hat = np.dot(train_set.T, w)
        check = y * y_hat

        while np.where(check <= 0)[0].shape[0] > 0:
            i = np.where(check <= 0)[0][0]
            w = w + (y[i] * train_set[:, i])

            y_hat = np.dot(train_set.T, w)
            check = y * y_hat

        self.model = w

    def predict(self, X):
        """
        predicts the label of a new data matrix of dim (d,m')
        :return: label vector of dim (m',)
        """
        X_intercept = np.insert(X, 0, values=np.full(X.shape[1], 1), axis=0)
        return np.sign(np.dot(X_intercept.T, self.model))


class LDA(model):
    def __init__(self):
        super().__init__()
        self.model = dict()

    def fit(self, X, y):
        """
        divide population into label 1 and -1, and compute estimators for each
        sub-sample mean. also, compute estimator for y probability (frequency to be
        labled 1 according to sample), and a pooled covariance matrix that considers
        two covariance matrices for each sub-sample.
        saves the results as a dict under self.model.
        :param X: train sample, matrix of dim (d,m)
        :param y: train label vector of dim (m,1)
        :return: None
        """
        positive = X.T[y == 1].T
        negative = X.T[y == -1].T
        self.model['y_pos_probability'] = np.mean(y == 1)
        self.model['mu_p_hat'] = np.mean(positive, axis=1)
        self.model['mu_n_hat'] = np.mean(negative, axis=1)
        self.model['cov_hat'] = (((positive.shape[1] - 1) * np.cov(positive) +
                                  (negative.shape[1] - 1) * np.cov(negative))
                                 / (X.shape[1] - 2))

    def __calc_discriminant(self, X, mu, inversed_cov, y_prob):
        """
        calculate the discriminant function using the given parameters.
        :return: double
        """
        return (X.T @ inversed_cov @ mu) - ((mu.T @ inversed_cov @ mu + np.log(y_prob))
                                            / 2)

    def predict(self, X):
        """
        returns for each sample the label that maximize the discriminant function.
        :param X: sample matrix, of dim (d, m')
        :return: label vector of dim (m',1)
        """
        inversed_cov = np.linalg.pinv(self.model['cov_hat'])
        mu_neg = self.model['mu_n_hat']
        mu_pos = self.model['mu_p_hat']
        y_pos_probability = self.model['y_pos_probability']

        delta_neg = self.__calc_discriminant(X, mu_neg, inversed_cov,
                                             1 - y_pos_probability)
        delta_pos = self.__calc_discriminant(X, mu_pos, inversed_cov, y_pos_probability)
        result = np.array(delta_pos > delta_neg, dtype=int)
        result[result == 0] = -1
        return result


class SVM(model):
    """
    Wrapper class for svm model for sklearn.
    """
    def __init__(self):
        super().__init__()
        self.model = SVC(C=1e10, kernel='linear')

    def fit(self, X, y):
        self.model.fit(X.T, y)

    def predict(self, X):
        return self.model.predict(X.T)


class Logistic(model):
    """
    Wrapper class for Logistic regression model for sklearn.
    """
    def __init__(self):
        super().__init__()
        self.model = LogisticRegression(solver='liblinear')

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class DecisionTree(model):
    """
    Wrapper class for Decision tree model for sklearn.
    """
    def __init__(self):
        super().__init__()
        self.model = DecisionTreeClassifier(max_depth=7)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
