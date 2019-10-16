import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_test, y_test = util.load_dataset(eval_path, add_intercept=True)
    
    #x_train[:,1] = np.log(x_train[:,1])
    #x_test[:,1] = np.log(x_test[:,1])
    
    # *** START CODE HERE ***
    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to pred_path
    
    
    gda = GDA()
    gda.fit(x_train, y_train)
    print('gda theta =', gda.theta)
    util.plot(x_test, y_test, gda.theta, 'gda.png')
    y_pred = gda.predict(x_test)
    np.savetxt(pred_path, y_pred)
    
    
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        # Write theta in terms of the parameters
        
        m, n = x.shape
        
        phi = y.mean()
        mu_0 = (1 - y) @ x / np.sum(1 - y)
        mu_1 = y @ x / y.sum()
        sigma = 1 / m * x.T @ x
        
        self.theta = np.zeros(n + 1)
        self.theta[0] = 1 / 2 * mu_1 @ np.linalg.inv(sigma) @ mu_1 - 1 / 2 * mu_0 @ np.linalg.inv(sigma) @ mu_0 + np.log(phi / (1 - phi))
        self.theta[1:] = (mu_0 - mu_1).T @ np.linalg.inv(sigma)

        
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
        h = 1 / (1 + np.exp(-z))
        
        return h
    
        # *** END CODE HERE