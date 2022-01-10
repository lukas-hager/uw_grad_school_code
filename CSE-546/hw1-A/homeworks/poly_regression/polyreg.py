# %%

"""
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

from typing import Tuple

import numpy as np
from scipy import linalg as la

from utils import problem

# %%
class PolynomialRegression:
    @problem.tag("hw1-A", start_line=4)
    def __init__(self, degree: int = 1, reg_lambda: float = 1e-8):
        """
        Constructor
        """
        self.degree: int = degree
        self.reg_lambda: float = reg_lambda
        self.std_vals = None

    @staticmethod
    @problem.tag("hw1-A")
    def polyfeatures(X: np.ndarray, degree: int) -> np.ndarray:
        """
        Expands the given X into an (n, degree) array of polynomial features of degree degree.

        Args:
            X (np.ndarray): Array of shape (n, 1).
            degree (int): Positive integer defining maximum power to include.

        Returns:
            np.ndarray: A (n, degree) numpy array, with each row comprising of
                X, X * X, X ** 3, ... up to the degree^th power of X.
                Note that the returned matrix will not include the zero-th power.

        """
        list = tuple([X ** power for power in range(1, degree + 1)])
        array = np.column_stack(list)
        return(array)

    @problem.tag("hw1-A")
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model, and saves learned weight in self.weight

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.
            y (np.ndarray): Array of shape (n, 1) with targets.

        Note:
            You need to apply polynomial expansion and scaling at first.
        """
        
        # convert to the polynomial form
        X_powers = self.polyfeatures(X, self.degree)

        # initialize dict for standardization
        std_vals = {}
        std_vals['mean'], std_vals['sd'] = [], []

        # standardize
        with np.nditer(
            X_powers, 
            flags=['external_loop'], 
            op_flags = ['readwrite'], 
            order='F'
        ) as it:
            for var in it:

                mean = var.mean()
                sd = var.std()

                std_vals['mean'] += [mean]
                std_vals['sd'] +=  [sd]

                var[...] = (var - mean) / sd

        # add a row of 1s to include the intercept
        X_mat = np.hstack(
            (np.ones(X.shape[0]).reshape((X.shape[0], 1)),
            X_powers)
        )
        
        # calculate beta as (lambda I + X'X)^-1X'Y; DON'T REGULARIZE OFFSET (first element)
        I_adj = np.identity(self.degree + 1)
        I_adj[0,0] = 0.
        beta = la.solve(
            self.reg_lambda * I_adj + X_mat.T @ X_mat,
            X_mat.T @ y
        )

        # save the weight and standardization
        self.weight = beta
        self.std_vals = std_vals

    @problem.tag("hw1-A")
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Use the trained model to predict values for each instance in X.

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.

        Returns:
            np.ndarray: Array of shape (n, 1) with predictions.
        """
        # convert to the polynomial form
        X_powers = self.polyfeatures(X, self.degree)

        # 
        X_powers = (X_powers - np.array(self.std_vals['mean'])) / np.array(self.std_vals['sd'])
        # add a row of 1s to include the intercept
        X_mat = np.hstack(
            (np.ones(X.shape[0]).reshape((X.shape[0], 1)),
            X_powers)
        )

        preds = X_mat @ self.weight

        return(preds)


@problem.tag("hw1-A")
def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    """Given two arrays: a and b, both of shape (n, 1) calculate a mean squared error.

    Args:
        a (np.ndarray): Array of shape (n, 1)
        b (np.ndarray): Array of shape (n, 1)

    Returns:
        float: mean squared error between a and b.
    """
    error = a - b 
    error_sq = error ** 2 
    mse = error_sq.mean()
    return(mse)

# %%
@problem.tag("hw1-A", start_line=5)
def learningCurve(
    Xtrain: np.ndarray,
    Ytrain: np.ndarray,
    Xtest: np.ndarray,
    Ytest: np.ndarray,
    reg_lambda: float,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute learning curves.

    Args:
        Xtrain (np.ndarray): Training observations, shape: (n, 1)
        Ytrain (np.ndarray): Training targets, shape: (n, 1)
        Xtest (np.ndarray): Testing observations, shape: (n, 1)
        Ytest (np.ndarray): Testing targets, shape: (n, 1)
        reg_lambda (float): Regularization factor
        degree (int): Polynomial degree

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            1. errorTrain -- errorTrain[i] is the training mean squared error using model trained by Xtrain[0:(i+1)]
            2. errorTest -- errorTest[i] is the testing mean squared error using model trained by Xtrain[0:(i+1)]

    Note:
        - For errorTrain[i] only calculate error on Xtrain[0:(i+1)], since this is the data used for training.
            THIS DOES NOT APPLY TO errorTest.
        - errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """
    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)

    # initialize the class
    pr = PolynomialRegression(
        degree = degree,
        reg_lambda = reg_lambda
    )

    # # assess the minimum amount of data required to compute the values
    # if reg_lambda > 0:
    #     min_data = 1
    # else:
    #     min_data = degree

    # Fill in errorTrain and errorTest arrays
    for i in range(1,n):
        
        # get the subset of data for training
        Xtrain_sub = Xtrain[0:(i+1)]
        Ytrain_sub = Ytrain[0:(i+1)]

        # fit to the subset of training data
        pr.fit(Xtrain_sub, Ytrain_sub)

        # use the parameters to predict
        Ypred_train = pr.predict(Xtrain_sub)
        Ypred_test = pr.predict(Xtest)

        # compute MSE
        mse_val_train = mean_squared_error(Ypred_train, Ytrain_sub)
        mse_val_test = mean_squared_error(Ypred_test, Ytest)

        # add to the vectors

        errorTrain[i] = mse_val_train
        errorTest[i] = mse_val_test
    
    return((errorTrain, errorTest))


