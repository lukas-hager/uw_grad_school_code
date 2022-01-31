# %%

import numpy as np
import pandas as pd
import scipy as sp
from scipy import optimize as op

data_og = pd.read_csv('/Users/hlukas/Google Drive/Grad School/2021-2022/Winter/ECON 532/Assignments/Assignment 1/pset1_upload/airline.txt')

data_og.columns = [x.lower() for x in data_og.columns.tolist()]

# import data, define indicator
data_binary = data_og.copy()
data_binary['late'] = data_binary['arr_delay'] > 15
data_binary['late'] = data_binary['late'].astype('float')
data_binary['cons'] = 1.

# convert to matrices

X = data_binary[['cons', 'distance', 'dep_delay']].to_numpy()
y = data_binary['late'].to_numpy()

# %% 
def get_ll(b_hat, X_val, y_val):
    exp_term = np.exp(X_val @ b_hat)
    prob_1 = exp_term / (1. + exp_term)
    prob_0 = 1 - prob_1
    return(-1 * np.sum(y_val.T @ np.log(prob_1) + (1.-y_val).T @ np.log(prob_0)))
# %%

min_vals = op.minimize(get_ll, np.array([1.,1.,1.]), args = (X,y), method = 'L-BFGS-B')
# %%
get_ll(min_vals['x'], X, y)
# %%
