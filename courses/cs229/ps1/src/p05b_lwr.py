import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    # Get MSE value on the validation set
    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data
    lwr = LocallyWeightedLinearRegression(tau)
    lwr.fit(x_train, y_train)
    y_pred = lwr.predict(x_val)

    plt.figure()
    plt.plot(x_train, y_train, 'ro', linewidth=2)
    plt.plot(x_val, y_pred, 'bo', linewidth=2)
    plt.plot(y_pred, y_val, 'gx', linewidth=1)

    # Add labels and save to disk
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('p05b.png')

    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***

        m, n = x.shape
        self.X = x
        self.Y = y

        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        mX, nX = self.X.shape
        y = np.zeros(m)
        W = np.zeros((m, mX))

        for i in range(m):
            for j in range(mX):
                W[i, j] = np.exp(- (x[i] - self.X[j]).T @ (
                            x[i] - self.X[j]) / 2 / self.tau ** 2)
            theta = np.linalg.inv(
                self.X.T @ np.diag(W[i]) @ self.X) @ self.X.T @ np.diag(
                W[i]) @ self.Y
            y[i] = x[i] @ theta

        return y

        # *** END CODE HERE ***
