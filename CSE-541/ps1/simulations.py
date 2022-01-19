# %%
import numpy as np
import numpy.random as rand
import pandas as pd

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

        cum_regret_val += (lever != 0) * 1
        cum_regret.append(cum_regret_val)

    
    initial_pulls = pd.DataFrame(data = {'lever': pulls, 'result': result})
    summary = initial_pulls.groupby('lever').agg(np.mean).reset_index()

    best_lever = summary['result'].argmax()

    for i in range(n*m,T):
        pulls.append(lever)
        result.append(rand.normal(loc = mu_i[lever], scale = v))

        cum_regret_val += (lever != 0) * 1
        cum_regret.append(cum_regret_val)

    final_df = pd.DataFrame(data = {'lever': pulls, 'result': result, 'cum_regret': cum_regret})

    return(final_df, best_lever)
            
final_df, best_lever = etc(5, 100, [1,0,0,0,0,0,0,0,0,0], 1)

# %%
pulls = []
result = []
mu_i = [1,0,0,0,0,0,0,0,0,0]

for i in range(10 * 3):
    lever = (i % 10)
    pulls.append(lever)
    result.append(rand.normal(loc = mu_i[lever], scale = 1))

initial_pulls = pd.DataFrame(data = {'lever': pulls, 'result': result})
summary = initial_pulls.groupby('lever').agg(np.mean).reset_index()

best_lever = summary['result'].argmax()

for i in range(m )
# %%
