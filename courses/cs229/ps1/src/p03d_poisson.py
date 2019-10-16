import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_test, y_test = util.load_dataset(eval_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path

    pr = PoissonRegression(step_size=lr)
    pr.fit(x_train, y_train)
    print(pr.theta)
    y_pred = pr.predict(x_test)
    np.savetxt(pred_path, y_pred)

    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***

        m, n = x.shape
        self.theta = np.zeros(n)
        error = 1e5
        n_iter = 0
        self.max_iter = 1e4

        while error > self.eps and n_iter < self.max_iter:
            dJ = (y - np.exp(x @ self.theta)) @ x / m
            J = (x @ self.theta * y - np.exp(
                x @ self.theta)).sum()
            self.theta += self.step_size * dJ
            error = np.linalg.norm(self.step_size * dJ, 1)
            n_iter += 1

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***

        z = x @ self.theta
        g = np.exp(z)

        return g
        # *** END CODE HERE ***
