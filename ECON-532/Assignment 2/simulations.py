# %%

import numpy as np
import numpy.random as rand
import pandas as pd
import scipy as sp
from scipy.io import loadmat
import matplotlib.pyplot as plt

# %%

mat_data = loadmat('/Users/hlukas/git/uw_grad_school_code/ECON-532/Assignment 2/pset2_upload/Simulated Data/100markets3products.mat')
#data_og = pd.DataFrame(data = {'{}'.format(x): mat_data[x] for x in list(mat_data.keys()) if x not in ['__header__', '__version__', '__globals__']},index=[0])
variable_names = [x for x in mat_data.keys() if x not in ['__header__', '__version__', '__globals__']]
x1,xi_all,w,eta,Z,alphas,P_opt,shares = [mat_data[x] for x in variable_names]
# %%

true_gamma = np.array([2,1,1]).reshape(3,1)
prods = 3
mkts = 100

mc_jm = np.column_stack([np.ones(prods * mkts),w, Z]) @ true_gamma + eta
markup = P_opt - mc_jm.reshape((3,100), order = 'F')

profit = markup * shares
# %%

fig, axs = plt.subplots(3)
fig.suptitle('Profit Distribution For Products Across Markets')

for i in range(3):
    axs[i].hist(
        markup[i,].squeeze(), 
        15, 
        range = (np.min(markup), np.max(markup)), 
        ec='black'
    )
fig.tight_layout()

# %%
fig, axs = plt.subplots(3)
fig.suptitle('Price Distribution For Products Across Markets')

for i in range(3):
    axs[i].hist(
        P_opt[i,].squeeze(), 
        15, 
        range = (np.min(markup), np.max(markup)), 
        ec='black'
    )
fig.tight_layout()
# %%

alpha = 1.
sigma_alpha = 1.
beta = np.array([5.,1.,1.]).reshape((3,1))

n_consumers = 1000

# simulate consumers
nu = rand.lognormal(size = n_consumers)

# get the alpha_i
alpha_i = alpha + sigma_alpha * nu

# get the market-product level utility

u_jm = (x1 @ beta + xi_all).reshape((3,100), order = 'F')
# %%

def wide_to_long(mat_val):
    


def long_to_wide(array_val):
    return(array_val.reshape((mkts,prods)))