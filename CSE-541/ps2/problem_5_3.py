# %%
from cProfile import label
import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt
import pickle
from os.path import exists

# %%

#### G-Optimal Design

def A(lambda_val,X):
    return((lambda_val.reshape(-1,1) * X).T @ X)

def h(A_val, X):
    A_val_inv = np.linalg.inv(A_val)
    vals = np.diag(X @ A_val_inv @ X.T)
    return(np.max(vals).item())

def g(A_val):
    return(-np.log(np.abs(np.linalg.det(A_val))))

def greedy(X, N):
    n,d = X.shape
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

def G_optimal(X_vals, N):   
    n,d = X_vals.shape
    I_t_sim = greedy(X_vals, N)
    lambda_hat = (np.array([np.sum(I_t_sim == i) for i in range(n)]) / N).squeeze()
    return(lambda_hat)

# %%

# taken from problem set

n=300

X = np.concatenate( (np.linspace(0,1,50), 0.25+ 0.01*np.random.randn(250) ), 0) 
X = np.sort(X)
K = np.zeros((n,n))

if exists('/Users/hlukas/git/uw_grad_school_code/CSE-541/ps2/phi.pkl'):
    with open('/Users/hlukas/git/uw_grad_school_code/CSE-541/ps2/phi.pkl', 'rb') as f:
        Phi = pickle.load(f)

else:
    for i in range(n):
        for j in range(n):
            K[i,j] = 1+min(X[i],X[j])
            e, v = np.linalg.eigh(K) # eigenvalues are increasing in order
            d = 30
            Phi = np.real(v @ np.diag(np.sqrt(np.abs(e))) )[:,(n-d)::]

    with open('/Users/hlukas/git/uw_grad_school_code/CSE-541/ps2/phi.pkl', 'wb') as f:
        pickle.dump(Phi, f)
# %%

def f(x):
    return -x**2 + x*np.cos(8*x) + np.sin(15*x)

f_star = f(X)
theta = np.linalg.lstsq( Phi, f_star, rcond=None)[0] 
f_hat = Phi @ theta

def observe(idx):
    return f(X[idx]) + np.random.randn(len(idx))

def sample_and_estimate(X, lbda, tau):
    n, d = X.shape
    reg = 1e-6 # we can add a bit of regularization to avoid divide by 0
    idx = np.random.choice(np.arange(n),size=tau,p=lbda) 
    y = observe(idx)
    XtX = X[idx].T @ X[idx]
    XtY = X[idx].T @ y
    theta = np.linalg.lstsq( XtX + reg*np.eye(d), XtY, rcond=None )[0]
    return Phi @ theta , XtX

T = 1000
lbda_G = G_optimal(Phi,1000)
f_G_Phi, A = sample_and_estimate(Phi, lbda_G, T)
conf_G = np.sqrt(np.sum(Phi @ np.linalg.inv(A) * Phi,axis=1))

lbda = np.ones(n)/n
f_unif_Phi, A = sample_and_estimate(Phi, lbda, T)
conf_unif = np.sqrt(np.sum(Phi @ np.linalg.inv(A) * Phi,axis=1))

# %%

plt.plot(np.arange(300.) / 299., np.cumsum(lbda_G), label ='dist_G')
plt.plot(np.arange(300.) / 299., np.cumsum([1./300.]*300), label = 'dist_unif')
plt.legend()
plt.xlabel('$x$')
plt.ylabel('CDF')
plt.title('CDF of G-Optimal Design and Uniform Distribution')
plt.savefig('/Users/hlukas/Google Drive/Grad School/2021-2022/Winter/CSE 541/Homework/Homework 2/5_3a.png')
plt.close()
# %%

plt.plot(np.arange(300.) / 299., f_G_Phi, label = 'f_G_Phi', ls='-.')
plt.plot(np.arange(300.) / 299., f_unif_Phi, label = 'f_unif_Phi', ls='-.')
plt.plot(np.arange(300.) / 299., f_star, label = 'f_star')
plt.legend()
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.title('Functional Approximation Using G-Optimal Design/Uniform')
plt.savefig('/Users/hlukas/Google Drive/Grad School/2021-2022/Winter/CSE 541/Homework/Homework 2/5_3b.png')
plt.close()
# %%
abs_diff_Phi = np.abs(f_star - f_G_Phi)
abs_diff_unif = np.abs(f_star - f_unif_Phi)
sqrt_d_n = np.repeat(np.sqrt(30. / 300), 300)

plt.plot(np.arange(300.) / 299., conf_unif, label = 'conf_unif', ls='--', color = 'b')
plt.plot(np.arange(300.) / 299., abs_diff_unif, label = 'diff_unif', color = 'b')
plt.plot(np.arange(300.) / 299., conf_G, label = 'conf_G', ls='--', color = 'r')
plt.plot(np.arange(300.) / 299., abs_diff_Phi, label = 'diff_G', color = 'r')
plt.plot(np.arange(300.) / 299., sqrt_d_n, label = 'sqrt_d_n')
plt.legend()
plt.xlabel('$x$')
plt.title('Confidence Intervals Using G-Optimal Design/Uniform')
plt.savefig('/Users/hlukas/Google Drive/Grad School/2021-2022/Winter/CSE 541/Homework/Homework 2/5_3c.png')
plt.close()