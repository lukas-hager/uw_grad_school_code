# %%

import numpy as np
import numpy.random as rand
import pandas as pd
import scipy as sp
from scipy.io import loadmat
import matplotlib.pyplot as plt

# %%

mat_data = loadmat('/Users/hlukas/git/uw_grad_school_code/ECON-532/Assignment 2/pset2_upload/Simulated Data/100markets3products.mat')
variable_names = [x for x in mat_data.keys() if x not in ['__header__', '__version__', '__globals__']]
x1,xi_all,w,eta,Z,alphas,P_opt,shares = [mat_data[x] for x in variable_names]
# %%

true_gamma = np.array([2,1,1]).reshape(3,1)
prods = 3
mkts = 100
n_consumers = 1000

mc_jm = np.column_stack([np.ones(prods * mkts),w, Z]) @ true_gamma + eta
markup = P_opt.T - mc_jm.reshape((mkts, prods))

profit = markup * shares.T * n_consumers
# %%

# fig, axs = plt.subplots(3)
# fig.suptitle('Profit Distribution For Products Across Markets')

# for i in range(3):
#     axs[i].hist(
#         markup[i,].squeeze(), 
#         15, 
#         range = (np.min(markup), np.max(markup)), 
#         ec='black'
#     )
# fig.tight_layout()

profit_flat = profit.reshape(prods * mkts)
plt.hist(profit_flat, bins=15, ec = 'black')
plt.title('Profit Distribution ({} Products, {} Markets, {} Consumers)'.format(prods, mkts, n_consumers))
plt.xlabel('Profit')
plt.ylabel('N')

# %%
# fig, axs = plt.subplots(3)
# fig.suptitle('Price Distribution For Products Across Markets')

# for i in range(3):
#     axs[i].hist(
#         P_opt[i,].squeeze(), 
#         15, 
#         range = (np.min(markup), np.max(markup)), 
#         ec='black'
#     )
# fig.tight_layout()
p_flat = P_opt.reshape(prods * mkts)
plt.hist(p_flat, bins=15, ec = 'black', range = (0, 7.5))
plt.title('Price Distribution ({} Products, {} Markets)'.format(prods, mkts))
plt.xlabel('Price')
plt.ylabel('N')
# %%

rand.seed(seed=2022)

alpha = 1.
sigma_alpha = 1.
beta = np.array([5.,1.,1.]).reshape((3,1))

# simulate consumers
nu = rand.lognormal(size = n_consumers)

# get the alpha_i
alpha_i = alpha + sigma_alpha * nu

# repeat the alpha_i for each market
alpha_col = np.concatenate([[x] * mkts for x in alpha_i]).reshape((mkts * n_consumers, 1))
alpha_big = np.tile(alpha_col, (1,3))

# get the market-product level utility

#u_jm = (x1 @ beta + xi_all).reshape((3,100), order = 'F')
u_jm = (x1 @ beta + xi_all).reshape((100,3))

# create a big system
u_big = np.tile(u_jm, (1000,1))
xi_big = np.tile(xi_all.reshape((100,3)), (1000,1))
p_big = np.tile(P_opt.T, (1000,1))
ep_big = rand.gumbel(size = mkts * prods * n_consumers).reshape((mkts * n_consumers, prods))

u = u_big + xi_big - alpha_big * p_big + ep_big
net_u = np.concatenate([u, np.zeros(mkts * n_consumers).reshape((mkts * n_consumers, 1))], axis = 1)
cs = net_u.max(axis = 1) / alpha_col.squeeze()

# %%

plt.hist(cs, bins = 20)
# %%

# def wide_to_long(mat_val):
    


# def long_to_wide(array_val):
#     return(array_val.reshape((mkts,prods)))