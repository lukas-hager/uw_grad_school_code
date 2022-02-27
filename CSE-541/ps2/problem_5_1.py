# %%
from cProfile import label
import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt
import pickle

# %%

delta = .05

def bound_all_t(sigma2,t):
    return(np.sqrt(1. + 1./(t*sigma2)) * np.sqrt((2*np.log(1/delta) + np.log(t * sigma2 + 1.)) / t))

# %%

def bound_fixed_t(t):
    return(np.sqrt(2 * np.log(2 / delta) / t))

# %%

def bound_ratio(sigma2,t):
    return(bound_all_t(sigma2,t) / bound_fixed_t(t))
# %% 

results_dict = {}

for sigma_exp in np.arange(-6.,1.,1.):
    print('Running exponent {}'.format(sigma_exp))
    sigma2 = 10 ** sigma_exp
    results_dict[sigma_exp] = bound_ratio(sigma2,np.arange(1.,10e6))

# %%
fig, axs = plt.subplots(7,sharex=True)
fig.suptitle('Confidence Bound Ratio for Different $\sigma^2$')
for i in range(7):  
    exp = list(results_dict.keys())[i]
    axs[i].plot(results_dict[exp])
    axs[i].axvline(x=results_dict[exp].argmin(),ls='--', color = 'r')
    axs[i].set_xscale('log')
    axs[i].axes.yaxis.set_ticklabels([])
    axs[i].axes.set_ylabel(int(exp))

fig.set_size_inches(5,8)
plt.savefig('/Users/hlukas/Google Drive/Grad School/2021-2022/Winter/CSE 541/Homework/Homework 2/5_1.png', dpi = 300)