import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    x_val, y_val = util.load_dataset(valid_path, label_col='t', add_intercept=True)
    
    lr_t = LogisticRegression()
    lr_t.fit(x_train, t_train)
    t_pred = lr_t.predict(x_test)
    np.savetxt(pred_path_c, t_pred)
    util.plot(x_test, t_test, lr_t.theta, 'p02c.png')
    
    lr_y = LogisticRegression()
    lr_y.fit(x_train, y_train)
    y_pred = lr_y.predict(x_test)
    np.savetxt(pred_path_d, y_pred)
    util.plot(x_test, t_test, lr_y.theta, 'p02d.png')
    
    y_pred_val = lr_y.predict(x_val)
    alpha = np.sum(y_pred_val[y_val == 1]) / y_val.sum()
    print('alpha = ', alpha)
    util.plot(x_test, t_test, lr_y.theta, 'p02e.png', correction=alpha)
    t_pred = lr_y.predict(x_test) / alpha
    np.savetxt(pred_path_e, t_pred)
    
    # *** END CODER HERE
