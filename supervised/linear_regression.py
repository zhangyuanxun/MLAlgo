import numpy as np
import math
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class LinearRegression(object):
    def __init__(self, learning_rate=0.01, num_iterations=100000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.train_errors = []

    def fit(self, X, y):
        num_features = X.shape[1]
        num_samples = X.shape[0]
        self.W = np.random.uniform(-1.0 / math.sqrt(num_features), 1.0 / math.sqrt(num_features), (num_features, ))
        self.b = 0

        for i in range(self.num_iterations):
            y_pred = np.dot(X, self.W) + self.b

            # compute mean square error
            mse = (0.5 / num_samples) * np.sum((y_pred - y) ** 2)
            self.train_errors.append(mse)
            print "iteration num: %d, mean square error: %f" % (i, mse)

            # compute gradient
            grad_w = (1.0 / num_samples) * ((y_pred - y).dot(X))
            grad_b = (1.0 / num_samples) * np.sum((y_pred - y))

            # update parameters
            self.W = self.W - self.learning_rate * grad_w
            self.b = self.b - self.learning_rate * grad_b

    def predict(self, X):
        return np.dot(X, self.W) + self.b


if __name__ == "__main__":

    # Load the diabetes dataset
    diabetes = datasets.load_diabetes()
    diabetes_X = diabetes.data[:, np.newaxis, 2]

    # split dataset
    diabetes_train_X, diabetes_test_X, diabetes_train_y, diabetes_test_y = train_test_split(diabetes_X,
                                                                                            diabetes.target,
                                                                                            test_size=0.2)

    # shuffle train dataset based on index
    s = np.arange(diabetes_train_X.shape[0])
    np.random.shuffle(s)
    diabetes_train_X = diabetes_train_X[s]
    diabetes_train_y = diabetes_train_y[s]

    linear_model = LinearRegression()
    linear_model.fit(diabetes_train_X, diabetes_train_y)

    # predict
    diabetes_y_predict = linear_model.predict(diabetes_test_X)

    # compute mse
    mse = (1.0 / diabetes_test_y.shape[0]) * np.sum((diabetes_y_predict - diabetes_test_y) ** 2)
    print("Mean squared error: %.2f" % mse)

    # Plot outputs
    plt.scatter(diabetes_test_X, diabetes_test_y,  color='black')
    plt.plot(diabetes_test_X, diabetes_y_predict, color='blue', linewidth=3)
    plt.xticks(())
    plt.yticks(())

    plt.show()
