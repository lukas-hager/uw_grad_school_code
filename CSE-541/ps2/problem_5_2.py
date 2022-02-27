# %%
from cProfile import label
import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt
import pickle
from os.path import exists

# %%

np.random.seed(99)

#### G-Optimal Design

def A(lambda_val,X):
    return((lambda_val.reshape(-1,1) * X).T @ X)

def h(A_val, X):
    A_val_inv = np.linalg.inv(A_val)
    vals = np.diag(X @ A_val_inv @ X.T)
    return(np.max(vals).item())

def g(A_val):
    return(-np.log(np.abs(np.linalg.det(A_val))))

def greedy(X, d, N):
    n = len(X)
    I_t = np.random.choice(np.arange(n),size=2*d)
    for round in range(2*d, N):
        f_vals = []
        mat_sum = X[I_t].T @ X[I_t]
        for i in range(n):
            total_mat = np.outer(X[i], X[i]) + mat_sum
            f_vals.append(g(total_mat))
        I_t = np.append(I_t,np.argmin(f_vals))
    return(I_t)

def generate_samples(a, n):
    mean = np.zeros(10)
    cov = np.diag([j ** (-a) for j in np.arange(1.,11.)])
    X = np.random.multivariate_normal(mean = mean, cov = cov, size = n)
    return(X)

# %%

if exists('/Users/hlukas/git/uw_grad_school_code/CSE-541/ps2/5_3_dict.pkl'):
    with open('/Users/hlukas/git/uw_grad_school_code/CSE-541/ps2/5_3_dict.pkl', 'rb') as f:
        results_dict = pickle.load(f)

else:
    N = 1000
    d = 10

    results_dict = {x: [] for x in [0., .5, 1., 2.]}

    for a in [0., .5, 1., 2.]:
        print(a)
        for n in [int(10. + 2. ** i) for i in np.arange(1.,11.)]:
            print(n)
            X_vals = generate_samples(a, n)
            I_t_sim = greedy(X_vals, d, N)
            lambda_hat = (np.array([np.sum(I_t_sim == i) for i in range(n)]) / N).reshape(-1,1)
            f_val = h(A(lambda_hat,X_vals), X_vals)
            results_dict[a].append(f_val)

    with open('/Users/hlukas/git/uw_grad_school_code/CSE-541/ps2/5_3_dict.pkl', 'wb') as f:
        pickle.dump(results_dict, f)
# %%

n_vals = [10. + 2. ** i for i in np.arange(1.,11.)]

for x in results_dict.keys():
    plt.plot(n_vals, results_dict[x], label = str(x), marker = 'o')
    plt.legend()
    plt.title('Greedy G-Optimal Design For Different Values of $a$ and $n$')
    plt.xlabel('$n$')
    #plt.xscale('log')
    plt.ylabel('$f(\hat{\lambda})$')

plt.savefig('/Users/hlukas/Google Drive/Grad School/2021-2022/Winter/CSE 541/Homework/Homework 2/5_2.png', dpi = 300)
plt.close()
# %%
