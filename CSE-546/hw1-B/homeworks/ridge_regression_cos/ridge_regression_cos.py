# %%

# Note for this import to work you need to call python from root directory.
from os import replace
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns

from homeworks.ridge_regression_mnist import ridge_regression
from utils import load_dataset, problem

RNG = np.random.RandomState(seed=546)

Dataset = Tuple[np.ndarray, np.ndarray]


@problem.tag("hw1-B")
def transform_data(x: np.ndarray, G: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Transform data according to the problem

    Args:
        x (np.ndarray): An array of shape (n, d). Observations.
        G (np.ndarray): Matrix G of shape (d, p), as specified in problem.
        b (np.ndarray): Array b of shape (p,), as specified in problem.

    Returns:
        np.ndarray: Cosine tranformation of input data x, after multiplication with G and addition of b.

    Note:
        - Do not call RNG in this function. G and b should be initialized in the main function.
            You will need to save them for part b.
    """

    # general dimensions: X is (n,d), G is (p,d), and b is (p,1). h(X) is (n,p).
    # to use matrix form, we use XG' + B', where B is n column vectors of b.
    n,d = x.shape
    B = np.hstack([b.reshape(len(b), 1)] * n)
    return(np.cos(x @ G + B.T))


@problem.tag("hw1-B")
def split_into_validation(
    x: np.ndarray, y: np.ndarray, fraction_train: float = 0.8
) -> Tuple[Dataset, Dataset]:
    """Splits training dataset into training and validation subsets.

    Args:
        x (np.ndarray): An array of shape (n, d). Observations of training set.
        y (np.ndarray): An array of shape (n,). Targets of training set.
        fraction_train (float, optional): Fraction of original (input) training, that should be kept in resulting training set.
            Remainder should go to validation. Defaults to 0.8.

    Returns:
        Tuple[Dataset, Dataset]: Tuple of two datasets. Each dataset is itself a tuple of 2 numpy array (observation and targets).
            Thus return should look similar to this: `return (x_train, y_train), (x_val, y_val)`
    """
    # get the total number of observations
    n_total = x.shape[0]

    # ceil to ensure an integer for the split
    n_train = int(np.floor(n_total * fraction_train))

    # get the indices randomly for train, and then the other elements for test
    indices_train = RNG.choice(np.arange(n_total), n_train, replace = False)
    indices_test = np.setxor1d(np.arange(n_total), indices_train)
    
    # define the arrays by the indices
    x_train,y_train = x[indices_train],y[indices_train]
    x_test,y_test = x[indices_test],y[indices_test] 

    return((x_train, y_train), (x_test, y_test))


@problem.tag("hw1-B", start_line=6)
def main():
    """Main function of the problem.
    It loads in data. You should perform a hyperparameter search over p, save the best performing weight, G and b.
    Then plots training and validation error as a function of p.
    Then, for the best p, report training, validation errors, as well test error with confidence interval around it.
    """
    # Load dataset and split train into train & validation
    (x, y), (x_test, y_test) = load_dataset("mnist")
    (x_train, y_train), (x_val, y_val) = split_into_validation(x, y, fraction_train=0.8)
    # Convert targets to one hot encoding
    y_train_one_hot = ridge_regression.one_hot(y_train, 10)
    ps = [10, 20, 40, 80, 160, 320, 640, 1000, 2000, 4000]  # Use these ps for search
    
    # get the dimension of the training data
    n,d = x_train.shape

    # initialize a list to save the loss
    loss_vals = {}
    loss_vals['mse_train'],loss_vals['mse_test'] = [], []

    # iterate over p values
    for p in ps:

        # create the random matrices
        G = RNG.normal(0, .1, (d,p))
        b = RNG.uniform(0,2*np.pi, (p,1))

        # create the cosine matrices (i.e. h(x))
        X_train = transform_data(x_train, G, b)
        X_test = transform_data(x_val, G, b)

        # train the model using training data
        W_hat = ridge_regression.train(X_train, y_train_one_hot, _lambda = 1e-4)

        # make predictions on both sets
        Y_pred_train = ridge_regression.predict(X_train, W_hat)
        Y_pred_test = ridge_regression.predict(X_test, W_hat)

        # get error

        train_mse = np.average(1 - np.equal(Y_pred_train, y_train)) * 100
        test_mse = np.average(1 - np.equal(Y_pred_test, y_val)) * 100

        # append to lists

        loss_vals['mse_train'].append(train_mse)
        loss_vals['mse_test'].append(test_mse)

        # print
        print("B2 Problem, p = {}".format(p))
        print(
            f"\tTrain Error: {train_mse:.2g}%"
        )
        print(f"\tTest Error:  {test_mse:.2g}%")

    # select best performing parameter
    best_p = ps[loss_vals['mse_test'].index(min(loss_vals['mse_test']))]

    print('The best-performing value of p is {}'.format(best_p))

    # get number of test obs

    m = x_test.shape[0]
    delta = .05
    conf_int = (np.log(2/delta) / (2 * m)) ** .5

    # print the confidence interval

    print('The 95 percent confidence interval is {0:.2f} p/m {1:.4f}'.format(
        min(loss_vals['mse_test']),
        conf_int)
    )

    # plot

    sns.set()

    plt.plot(ps, loss_vals['mse_train'], 'red', label = 'Training Loss', marker = 'o')
    plt.plot(ps, loss_vals['mse_test'], 'green', label = 'Testing Loss', marker = 'o')
    plt.xlabel('p')
    plt.ylabel('Accuracy')
    y_labs = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{}%'.format(lab) for lab in y_labs])

    plt.legend(
        loc="upper right", 
        #bbox_to_anchor=(1.1, 0.2), 
        fancybox=True, 
        shadow=True, 
        ncol=1
    )

    plt.show()

# %%
if __name__ == "__main__":
    main()
