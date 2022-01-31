# %%
import numpy as np
import numpy.random as rand
import pandas as pd
import matplotlib.pyplot as plt

# %%
def etc(m, T, mu_i, v):

    n = len(mu_i)
    pulls = []
    result = []
    cum_regret = []
    cum_regret_val = 0.

    for i in range(n * m):
        lever = (i % n)
        pulls.append(lever)
        result.append(rand.normal(loc = mu_i[lever], scale = v))

        cum_regret_val += 1. - mu_i[lever]
        cum_regret.append(cum_regret_val)

    
    initial_pulls = pd.DataFrame(data = {'lever': pulls, 'result': result})
    summary = initial_pulls.groupby('lever').agg(np.mean).reset_index()

    best_lever = summary['result'].argmax()

    for i in range(n*m,T):
        pulls.append(best_lever)
        result.append(rand.normal(loc = mu_i[best_lever], scale = v))

        cum_regret_val += 1. - mu_i[best_lever]
        cum_regret.append(cum_regret_val)

    final_df = pd.DataFrame(data = {
        'lever': pulls, 
        'result': result, 
        'cum_regret': cum_regret
    })

    return(final_df)
            
# %%

def ucb(T,mu_i,v):

    n = len(mu_i)
    pulls = []
    result = []
    cum_regret = []
    cum_regret_val = 0.

    for i in range(n):
        lever = (i % n)
        pulls.append(lever)
        result.append(rand.normal(loc = mu_i[lever], scale = v))

        cum_regret_val += (lever != 0) * 1
        cum_regret.append(cum_regret_val)

    for i in range(n,T):
        initial_pulls = pd.DataFrame(data = {'lever': pulls, 'result': result})
        summary = initial_pulls.groupby(
            'lever', 
            group_keys=False
        ).agg(
            ['mean', 'count']
        ).droplevel(
            level = 0, axis = 1
        ).reset_index()
        summary['ucb']=summary['mean']+(2*np.log(2*n*T**2)/summary['count'])**.5

        best_lever = summary['ucb'].argmax()

        pulls.append(best_lever)
        result.append(rand.normal(loc = mu_i[best_lever], scale = v))

        cum_regret_val += 1. - mu_i[best_lever]
        cum_regret.append(cum_regret_val)

    final_df = pd.DataFrame(data = {
        'lever': pulls, 
        'result': result, 
        'cum_regret': cum_regret
    })

    return(final_df)

# %%

# P(mu|X) = P(X|mu) * P(mu) / P(X)
# X ~ N(mu,v)
# mu ~ N(0,1)

def thompson(T,mu_i,v_i):

    n = len(mu_i)
    pulls = []
    result = []
    cum_regret = []
    cum_regret_val = 0.
    all_arms = pd.DataFrame(data = {
        'lever': np.arange(len(mu_i))
    }).set_index('lever')

    mu_post = np.zeros(len(mu_i))
    v_post = np.ones(len(v_i))

    for i in range(T):
        theta_draws = rand.normal(mu_post, v_post, len(mu_i))
        best_lever = theta_draws.argmax()
        pulls.append(best_lever)

        result.append(rand.normal(mu_i[best_lever], v_i[best_lever], 1).item())

        all_pulls = pd.DataFrame(data = {'lever': pulls, 'result': result})
        summary = all_pulls.groupby(
            'lever', 
            group_keys=False
        ).agg(
            ['sum', 'count']
        ).droplevel(
            level = 0, 
            axis = 1
        ).reset_index()
        summary['mu_post'] = summary['sum'] / (summary['count'] + 1)
        summary['v_post'] = 1. / (summary['count'] + 1)
        summary = summary.set_index(
            'lever'
        ).join(
            all_arms, how = 'right'
        ).reset_index()
        summary['mu_prior'] = np.zeros(len(mu_i))
        summary['v_prior'] = np.ones(len(v_i))
        summary['mu_post']=summary['mu_post'].combine_first(summary['mu_prior'])
        summary['v_post']=summary['v_post'].combine_first(summary['v_prior'])
        summary.sort_values(by = 'lever', inplace = True)

        #print(summary)

        mu_post = summary['mu_post']
        v_post = summary['v_post']

        # print(mu_post)
        # print(v_post)

        cum_regret_val += 1. - mu_i[best_lever]
        cum_regret.append(cum_regret_val)

    final_df = pd.DataFrame(data = {
        'lever': pulls, 
        'result': result, 
        'cum_regret': cum_regret
    })

    return(final_df)

# %%

def plot_regret(T, mu_i, v_i, subpart):
    np.random.seed(2022)

    etc_df_1 = etc(1, T, mu_i, 1)
    etc_df_5 = etc(5, T, mu_i, 1)
    etc_df_10 = etc(10, T, mu_i, 1)
    ucb_df = ucb(T, mu_i, 1)
    thompson_df = thompson(T, mu_i, v_i)

    names = ['ETC (M=1)', 'ETC (M=5)', 'ETC (M=10)', 'UCB', 'Thompson']
    algos = [etc_df_1, etc_df_5, etc_df_10, ucb_df, thompson_df]

    plt.figure()

    for i,df in enumerate(algos):
        plt.plot(df['cum_regret'], label = names[i])

    plt.legend()
    plt.title('Regret of Algorithms, T = {}'.format(T))
    plt.xlabel('T')
    plt.ylabel('Cumulative Regret')
    plt.savefig('/Users/hlukas/Google Drive/Grad School/2021-2022/Winter/CSE 541/Homework/Homework 1/4_{}.png'.format(subpart))
    plt.close()

# %%

plot_regret(
    T = 1000, 
    mu_i = np.concatenate([np.ones(1), np.zeros(9)]), 
    v_i = np.ones(10),
    subpart=1
)

plot_regret(
    T = 10000,
    mu_i = np.concatenate(
        [np.ones(1), 1.-(1./np.sqrt(np.arange(2.,41.) - 1.))]
    ),
    v_i = np.ones(40),
    subpart=2
)
