import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_test, y_test = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to pred_path
        
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    util.plot(x_test, y_test, lr.theta, 'lr.png')
    y_pred = lr.predict(x_test)
    np.savetxt(pred_path, y_pred)
    print(lr.theta)
    
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        
        m, n = x.shape
        self.theta = np.zeros(n)
        error = 1e5
        n_iter = 0
        
        while error > self.eps and n_iter < self.max_iter:
            z = x @ self.theta
            g = 1 / (1 + np.exp(-z))
            diff = -1 / m * (y - g) @ x
            H = 1 / m * x.T @ np.diag(g * (1 - g)) @ x
            delta = np.linalg.inv(H) @ diff
            self.theta -= delta
            error = np.sum(np.abs(delta))
            n_iter += 1
            
           
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        z = x @ self.theta
        g = 1 / (1 + np.exp(-z))
        
        return(g)
        
        # *** END CODE HERE ***
