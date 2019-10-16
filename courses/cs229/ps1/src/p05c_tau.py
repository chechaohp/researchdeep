import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_val, y_val = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    
    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    # Fit a LWR model with the best tau value
    # Run on the test set to get the MSE value
    # Save predictions to pred_path
    # Plot data
    
    m, n = x_val.shape
    mse = np.zeros_like(tau_values)
    i = 0
    
    for tau in tau_values:
        lwr = LocallyWeightedLinearRegression(tau)
        lwr.fit(x_train, y_train)
        y_pred = lwr.predict(x_val)

        mse[i] = 1 / m * np.sum((y_pred - y_val) ** 2)

        i += 1
    
    print('tau=', tau_values)
    print('mse=', mse)
    
    i = np.argmin(tau_values)
    tau = tau_values[i]
    lwr = LocallyWeightedLinearRegression(tau)
    lwr.fit(x_train, y_train)
    y_pred = lwr.predict(x_test)
    np.savetxt(pred_path, y_pred)
    
    plt.figure()
    plt.plot(x_train, y_train, 'ro', linewidth=2)
    plt.plot(x_test, y_pred, 'bo', linewidth=2)
    plt.plot(x_test, y_test, 'gx', linewidth=1)
    
    # Add labels and save to disk
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('p05c.png')
    
    plt.figure()
    plt.plot(y_test, y_pred, 'ro')
    plt.savefig('test.png')
    
    # *** END CODE HERE ***