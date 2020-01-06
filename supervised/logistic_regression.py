from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import math
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


class LogisticRegression(object):
    def __init__(self, learning_rate=0.001, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.training_error = []

    def fit(self, X, y):
        num_samples = X.shape[0]
        num_features = X.shape[1]

        # initialize weight
        self.W = np.random.uniform(-1.0 / math.sqrt(num_features), 1.0 / math.sqrt(num_features), (num_features, ));
        self.b = 0

        for i in range(self.num_iterations):
            z = np.dot(X, self.W) + self.b
            y_pred = sigmoid(z)

            # compute cost using cross-entropy
            cost = np.multiply(-y, np.log(y_pred)) + np.multiply(-(1 - y), np.log(1 - y_pred))
            cost = (1.0 / num_samples) * np.sum(cost)

            self.training_error.append(cost)
            print "iteration num: %d, cost is: %f" % (i, cost)

            # compute gradient
            grad_w = (1.0 / num_samples) * (y_pred - y).dot(X)
            grad_b = (1.0 / num_samples) * np.sum((y_pred - y))

            self.W = self.W - self.learning_rate * grad_w
            self.b = self.b - self.learning_rate * grad_b

    def predict(self, X):
        y_pred = sigmoid(np.dot(X, self.W) + self.b)
        y_pred[np.where(y_pred >= 0.5)] = 1
        y_pred[np.where(y_pred < 0.5)] = 0

        return y_pred


if __name__ == "__main__":

    # load iris dataset
    iris = load_iris()
    iris_X, iris_y = iris.data, iris.target

    # remove last class
    iris_X = iris_X[:100]
    iris_y = iris_y[:100]

    # shuffle dataset
    s = np.arange(iris_y.shape[0])
    np.random.shuffle(s)
    iris_X = iris_X[s]
    iris_y = iris_y[s]

    print iris_y

    # split dataset
    iris_X_train, iris_X_test, iris_y_train, iris_y_test = train_test_split(iris_X, iris_y, test_size=0.2)

    # train model
    logistic_regression = LogisticRegression()
    logistic_regression.fit(iris_X_train, iris_y_train)

    # compute train accuracy
    y_pred = logistic_regression.predict(iris_X_train)
    accuracy = accuracy_score(iris_y_train, y_pred)
    print("Train Accuracy for %0.2f%% " % (accuracy * 100))

    # plot training loss
    plt.plot(range(1, len(logistic_regression.training_error) + 1), logistic_regression.training_error, color='blue',
             linewidth=3)
    plt.show()

    # test prediction
    y_pred = logistic_regression.predict(iris_X_test)

    # compute test accuracy
    accuracy = accuracy_score(iris_y_test, y_pred)
    print("Test Accuracy for %0.2f%% " % (accuracy * 100))
