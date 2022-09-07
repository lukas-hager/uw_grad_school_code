# %%

from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
import pickle
# %%

run_algos = False

def calculate_eigen(demean_data):
    return(np.linalg.eig(demean_data.T @ demean_data / len(demean_data)))

def project(demean_data, eigenvectors, d):
    proj_mat = eigenvectors[:,:d]
    return(demean_data @ proj_mat)

# %%
mndata = MNIST(
    '/Users/hlukas/git/uw_grad_school_code/CSE-541/ps3/data/'
)
images, labels = mndata.load_training()

# %%
images_mat = np.array(images)
labels_vec = np.array(labels)

# %%
def downsample(num):
    return(
        np.random.choice(
            np.array(np.where(labels_vec == num)).reshape(-1), 
            size = 5000, 
            replace = False
        )
    )


# %%

def phi(c,i):
    e_i = np.zeros(10)
    e_i[i] = 1.
    return((c.reshape(-1,1) @ e_i.reshape(1,-1)).reshape(-1))

def etc_world(tau):
    r_t = []
    a_t = []
    phi_t = []
    for i in range(tau):
        a = np.random.choice(10)
        r_t.append(a == labels_sample[i])
        phi_t.append(phi(proj_data[i], a))
        a_t.append(a)
    
    Phi = np.row_stack(phi_t)
    R = np.array(r_t) * 1.
    theta_hat = (np.linalg.inv(Phi.T @ Phi) @ Phi.T @ R).reshape(-1)
    
    for i in range(tau, 50000):
        a = np.argmax([np.inner(phi(proj_data[i], a_val), theta_hat) for a_val in range(10)])
        r_t.append(a == labels_sample[i])
        a_t.append(a)
    return(r_t)

def etc_bias(tau):
    r_t = []
    a_t = []
    c_t = []
    for i in range(tau):
        a = np.random.choice(10)
        r_t.append(a == labels_sample[i])
        c_t.append(proj_data[i])
        a_t.append(a)
    
    C = proj_data[:tau]
    A = np.array(a_t)
    logreg = LogisticRegression(max_iter=1e10)
    logreg.fit(C[r_t], A[r_t])
    
    a = logreg.predict(proj_data[tau:])
    r_t+=(a == labels_sample[tau:]).tolist()
    a_t+=a.tolist()
    return(r_t)

def sher_mor(mat,ob):
    num = mat @ ob @ ob.T @ mat 
    den = (1. + ob.T @ mat @ ob)
    return(mat - num / den)

def ftl(tau):
    r_t = []
    a_t = []
    phi_t = []
    for i in range(tau):
        a = np.random.choice(10)
        r_t.append(a == labels_sample[i])
        phi_t.append(phi(proj_data[i], a))
        a_t.append(a)

    Phi = np.row_stack(phi_t)
    inv_term = np.linalg.inv(Phi.T @ Phi)
    
    for i in range(tau, 50000):
        
        Phi = np.row_stack(phi_t)
        R = np.array(r_t) * 1.
        theta_hat = (inv_term @ Phi.T @ R).reshape(-1)
        
        a = np.argmax([np.inner(phi(proj_data[i], a_val), theta_hat) for a_val in range(10)])
        phi_term = phi(proj_data[i], a)
        r_t.append(a == labels_sample[i])
        phi_t.append(phi_term)
        a_t.append(a)
        inv_term = sher_mor(inv_term, phi_term.reshape(-1,1))
    return(r_t)

def UCB():

    T = 50000
    delta = 1/T
    gamma = 1
    U = 1
    V_0 = gamma * np.eye(d*10)
    V_t = V_0
    V_t_inv = np.linalg.inv(V_t)
    S_t = np.zeros(d*10)
    y_all = []

    for round in range(T):
        beta_t = np.sqrt(gamma) * U + np.sqrt(2 * np.log(1/delta) + np.linalg.slogdet(V_t)[1] - np.linalg.slogdet(V_0)[1])
        theta_t = V_t_inv @ S_t
        Phi = np.row_stack([phi(proj_data[round], x) for x in range(10)])
        x_t = np.argmax(Phi @ theta_t + beta_t * np.diag(Phi @ V_t_inv @ Phi.T))
        Phi_t = phi(proj_data[round], x_t)
        y_t = 1. * (x_t == labels_sample[round])
        V_t = V_t + np.outer(Phi_t,Phi_t)
        V_t_inv = sher_mor(V_t_inv, Phi_t.reshape(-1,1))
        S_t = S_t + Phi_t*y_t
        y_all.append(y_t)

    return(y_all)

def Thompson():

    T = 50000
    gamma = 1
    V_0 = gamma * np.eye(d*10)
    V_t = V_0
    V_t_inv = np.linalg.inv(V_t)
    S_t = np.zeros(d*10)
    y_all = []

    for round in range(T):
        theta_t = V_t_inv @ S_t
        theta_tilde_t = np.random.multivariate_normal(mean = theta_t, cov = V_t_inv)
        Phi = np.row_stack([phi(proj_data[round], x) for x in range(10)])
        x_t = np.argmax(Phi @ theta_tilde_t)
        Phi_t = Phi[x_t]
        y_t = 1. * (x_t == labels_sample[round])
        V_t_inv = sher_mor(V_t_inv, Phi_t.reshape(-1,1))
        S_t = S_t + Phi_t*y_t
        y_all.append(y_t)

    return(y_all)

if run_algos == True:
    results_etc_world = []
    results_etc_bias = []
    results_ftl = []
    results_ucb = []
    results_thompson = []

    for i in range(10):

        sample = np.concatenate([downsample(x) for x in range(10)])
        np.random.shuffle(sample)
        images_sample = images_mat[sample]
        labels_sample = labels_vec[sample]

        images_demean = images_sample - np.mean(images_sample, axis = 0)
        images_norm = images_demean / np.sqrt(np.sum(images_demean ** 2, axis = 1)).reshape(-1,1)
        
        d = 20

        vals, eigs = calculate_eigen(images_norm)
        proj_data = project(images_norm, np.real(eigs), d)

        big_Phi = np.row_stack([np.row_stack([phi(proj_data[c],x) for x in range(10)]) for c in range(50000)])
        big_R = np.zeros(10 * 50000)
        big_R[np.array([int(labels_sample[i] + i * 10) for i in range(50000)])] = 1.
        
        best_theta = np.linalg.inv(big_Phi.T @ big_Phi) @ big_Phi.T @ big_R
        best_lin_preds = np.array([np.argmax(big_Phi[(10*i):(10*(i+1))] @ best_theta) for i in range(50000)])
        best_lin_reward = (best_lin_preds == labels_sample) * 1.

        logreg = LogisticRegression(max_iter=1e10)
        logreg.fit(proj_data, labels_sample)

        best_logreg_preds = logreg.predict(proj_data)
        best_logreg_reward = (best_logreg_preds == labels_sample) * 1.

        print('ETC World: {}'.format(i))
        rewards = etc_world(15000)
        diff = best_logreg_reward - np.array(rewards)
        cum_sum = np.cumsum(diff)
        results_etc_world.append(cum_sum)

        print('ETC Bias: {}'.format(i))
        rewards = etc_bias(15000)
        diff = best_logreg_reward - np.array(rewards)
        cum_sum = np.cumsum(diff)
        results_etc_bias.append(cum_sum)

        print('FTL: {}'.format(i))
        rewards = ftl(10000)
        diff = best_logreg_reward - np.array(rewards)
        cum_sum = np.cumsum(diff)
        results_ftl.append(cum_sum)

        print('UCB: {}'.format(i))
        rewards = UCB()
        diff = best_logreg_reward - np.array(rewards)
        cum_sum = np.cumsum(diff)
        results_ucb.append(cum_sum)

        print('Thompson: {}'.format(i))
        rewards = Thompson()
        diff = best_logreg_reward - np.array(rewards)
        cum_sum = np.cumsum(diff)
        results_thompson.append(cum_sum)

        with open('/Users/hlukas/git/uw_grad_school_code/CSE-541/ps3/etc_world.pkl', 'wb') as f:
            pickle.dump(results_etc_world, f)

        with open('/Users/hlukas/git/uw_grad_school_code/CSE-541/ps3/etc_bias.pkl', 'wb') as f:
            pickle.dump(results_etc_bias, f)

        with open('/Users/hlukas/git/uw_grad_school_code/CSE-541/ps3/ftl.pkl', 'wb') as f:
            pickle.dump(results_ftl, f)

        with open('/Users/hlukas/git/uw_grad_school_code/CSE-541/ps3/ucb.pkl', 'wb') as f:
            pickle.dump(results_ucb, f)

        with open('/Users/hlukas/git/uw_grad_school_code/CSE-541/ps3/thompson.pkl', 'wb') as f:
            pickle.dump(results_thompson, f)
# %%

file_names = ['etc_world_2', 'etc_bias', 'ftl', 'ucb', 'thompson']
algo_names = ['ETC (World)', 'ETC (Bias)', 'FTL', 'UCB', 'Thompson']
colors = ['b', 'r', 'g', 'c', 'm']
for i in range(len(file_names)):
    file_name = file_names[i]
    algo_name = algo_names[i]
    color = colors[i]
    with open('/Users/hlukas/git/uw_grad_school_code/CSE-541/ps3/{}.pkl'.format(file_name), 'rb') as f:
        results = pickle.load(f)
    
    results_mat = np.row_stack(results)
    perc_90 = np.percentile(results_mat, 90, axis = 0)
    perc_10 = np.percentile(results_mat, 10, axis = 0)
    mean_val = np.mean(results_mat, axis = 0)
    plt.plot(mean_val, color = color, label = algo_name)
    plt.plot(perc_90, color = color, alpha = .1, ls = '--')
    plt.plot(perc_10, color = color, alpha = .1, ls = '--')

plt.legend()
plt.title('MNIST Regret')
plt.xlabel('$n$')
plt.ylabel('Regret')
plt.savefig('/Users/hlukas/git/uw_grad_school_code/CSE-541/ps3/2.png')
