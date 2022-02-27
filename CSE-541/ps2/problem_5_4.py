# %%
from cProfile import label
import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt
import pickle
from os.path import exists

# %%

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

def Eliminator():

    T = 40000
    tau = 100
    delta = 1/T
    gamma = 1
    U = 1
    V_0 = gamma * np.eye(30)
    V_t = V_0
    S_t = np.zeros(30).reshape(-1,1)
    chi_t = Phi
    y_all = []

    for round in range(int(T/tau)):
        lambda_k = G_optimal(chi_t,200)
        idx = np.random.choice(np.arange(len(chi_t)),size=tau,p=lambda_k) 
        X_t = chi_t[idx]
        y_t = X_t @ theta + np.random.randn(tau)
        V_t = V_t + X_t.T @ X_t
        S_t = S_t + np.sum(X_t * y_t.reshape(-1,1), axis = 0).reshape(-1,1)
        theta_t = np.linalg.inv(V_t) @ S_t 
        beta_t = np.sqrt(gamma) * U + np.sqrt(2 * np.log(1/delta) + np.log(np.linalg.det(V_t) / np.linalg.det(V_0)))
        x_hat_t_ind = np.argmax(chi_t @ theta_t)
        x_hat_t = chi_t[x_hat_t_ind]
        diff_mat = np.repeat(x_hat_t.reshape(1,-1), len(chi_t), axis = 0) - chi_t
        test_vals = diff_mat @ theta_t > (beta_t * np.sqrt(np.diag(diff_mat @ np.linalg.inv(V_t) @ diff_mat.T))).reshape(-1,1)
        chi_t = chi_t[test_vals.squeeze() == False]
        y_all.append(y_t)
    
    best_arm = np.max(Phi @ theta)
    regret = np.cumsum(np.repeat(best_arm, T) - np.array(y_all).reshape(-1,1).squeeze())
    return(regret)


# %%

######## UCB

def UCB():

    T = 40000
    delta = 1/T
    gamma = 1
    U = 1
    V_0 = gamma * np.eye(30)
    V_t = V_0
    S_t = np.zeros(30)
    y_all = []

    for round in range(T):
        beta_t = np.sqrt(gamma) * U + np.sqrt(2 * np.log(1/delta) + np.log(np.linalg.det(V_t) / np.linalg.det(V_0)))
        theta_t = np.linalg.inv(V_t) @ S_t
        x_t_ind = np.argmax(Phi @ theta_t + beta_t * np.sqrt(np.diag(Phi @ np.linalg.inv(V_t) @ Phi.T)))
        x_t = Phi[x_t_ind]
        y_t = np.inner(x_t,theta) + np.random.randn(1)
        V_t = V_t + np.outer(x_t,x_t)
        S_t = S_t + x_t*y_t
        y_all.append(y_t)

    best_arm = np.max(Phi @ theta)
    regret = np.cumsum(np.repeat(best_arm, T) - np.array(y_all).squeeze())
    return(regret)
# %%

#################### Thompson

def Thompson():

    T = 40000
    gamma = 1
    U = 1
    V_0 = gamma * np.eye(30)
    V_t = V_0
    S_t = np.zeros(30)
    y_all = []

    for round in range(T):
        theta_t = np.linalg.inv(V_t) @ S_t
        theta_tilde_t = np.random.multivariate_normal(mean = theta_t, cov = np.linalg.inv(V_t))
        x_t_ind = np.argmax(Phi @ theta_tilde_t)
        x_t = Phi[x_t_ind]
        y_t = np.inner(x_t,theta) + np.random.randn(1)
        V_t = V_t + np.outer(x_t,x_t)
        S_t = S_t + x_t*y_t
        y_all.append(y_t)

    best_arm = np.max(Phi @ theta)
    regret = np.cumsum(np.repeat(best_arm, T) - np.array(y_all).squeeze())
    return(regret)

def algo_plots(fcn):
    outcomes = []
    for i in range(10):
        outcomes.append(fcn())
    all_runs = np.row_stack(outcomes)
    return(np.mean(all_runs,axis = 0), np.min(all_runs, axis = 0), np.max(all_runs,axis = 0))
    #return(all_runs)

# %%
mean_val_t, min_val_t, max_val_t = algo_plots(Thompson)
mean_val_elim, min_val_elim, max_val_elim = algo_plots(Eliminator)
mean_val, min_val, max_val = algo_plots(UCB)

# %%
plt.plot(mean_val, color = 'b', label = 'UCB')
plt.plot(min_val, color = 'b', alpha = .1, ls = '--')
plt.plot(max_val, color = 'b', alpha = .1, ls = '--')
plt.plot(mean_val_t, color = 'r', label = 'Thompson')
plt.plot(min_val_t, color = 'r', alpha = .1, ls = '--')
plt.plot(max_val_t, color = 'r', alpha = .1, ls = '--')
plt.plot(mean_val_elim, color = 'g', label = 'Eliminator')
plt.plot(min_val_elim, color = 'g', alpha = .1, ls = '--')
plt.plot(max_val_elim, color = 'g', alpha = .1, ls = '--')
plt.title('Expected Regret for Different Algorithms')
plt.xlabel('T')
plt.ylabel('Expected Regret')
plt.legend()
plt.savefig('/Users/hlukas/Google Drive/Grad School/2021-2022/Winter/CSE 541/Homework/Homework 2/5_4.png')
plt.close()