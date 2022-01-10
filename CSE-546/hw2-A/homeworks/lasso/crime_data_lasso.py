# %%

if __name__ == "__main__":
    from coordinate_descent_algo import train  # type: ignore
else:
    from .coordinate_descent_algo import train

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem

# %%
def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    """Given two arrays: a and b, both of shape (n, 1) calculate a mean squared error.

    Args:
        a (np.ndarray): Array of shape (n, 1)
        b (np.ndarray): Array of shape (n, 1)

    Returns:
        float: mean squared error between a and b.
    """
    error = a - b 
    error_sq = error ** 2 
    mse = error_sq.mean()
    return(mse)

@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    
    df_train, df_test = load_dataset("crime")

    train_targets, test_targets = df_train['ViolentCrimesPerPop'].values, df_test['ViolentCrimesPerPop'].values
    train_obs = df_train.drop(['ViolentCrimesPerPop'], axis = 1).values
    test_obs = df_test.drop(['ViolentCrimesPerPop'], axis = 1).values

    # max lambda
    y_demean = train_targets - train_targets.mean()
    lambda_max = 2 * np.abs((train_obs.T * y_demean).sum(axis = 1)).max()

    # get the variables for part d and indices

    colnames = df_train.drop(['ViolentCrimesPerPop'], axis = 1).columns.tolist()
    vars = ['agePct12t29','pctWSocSec','pctUrban','agePct65up','householdsize']
    vars_idx = [colnames.index(x) for x in vars]

    sets = ['train', 'test']

    # initialize for loops
    lambda_val = lambda_max
    w = np.zeros(train_obs.shape[1])
    weight_dict = {var: [] for var in vars}
    error_dict = {var: [] for var in sets}

    lambda_list = []
    nonzero_weight_list = []

    # run the loops
    while lambda_val > .01:
        weights,bias = train(
            X = train_obs, 
            y = train_targets, 
            _lambda = lambda_val,
            start_weight=w
        )   

        lambda_list.append(lambda_val)
        nonzero_weight_list.append((np.abs(weights) > 0.).sum())

        for i,var in enumerate(vars):
            weight_dict[var].append(weights[vars_idx[i]])

        error_dict['train'].append(
            mean_squared_error(
                train_targets,
                train_obs @ weights + bias
            )
        )

        error_dict['test'].append(
            mean_squared_error(
                test_targets,
                test_obs @ weights + bias
            )
        )

        #w = np.copy(weights)
        lambda_val = np.copy(lambda_val) / 2

    weights_30,bias_30 = train(
        X = train_obs, 
        y = train_targets, 
        _lambda = 30,
        start_weight=np.zeros(train_obs.shape[1])
    )   

    weights_30_list = weights_30.tolist()

    max_30_idx = weights_30_list.index(weights_30.max())
    min_30_idx = weights_30_list.index(weights_30.min())

    # output plot c
    plt.plot(lambda_list, nonzero_weight_list, marker = 'o')
    plt.xscale('log')
    plt.title('Nonzero Weights per Regularization Value')
    plt.xlabel('Lambda')
    plt.ylabel('Nonzero Coefficients')
    plt.savefig('/Users/hlukas/Google Drive/Grad School/2021-2022/Autumn/CSE 546/Assignments/HW2/A3c.png')
    plt.close()

    # output plot d
    for var in vars:
        plt.plot(lambda_list, weight_dict[var], label = var, marker = 'o')
    plt.xscale('log')
    plt.title('Variable Weights per Regularization Value')
    plt.xlabel('Lambda')
    plt.ylabel('Weight')
    plt.legend(
        loc="lower right", 
        bbox_to_anchor=(1.1, 0.2), 
        fancybox=True, 
        shadow=True, 
        ncol=1
    )
    plt.savefig('/Users/hlukas/Google Drive/Grad School/2021-2022/Autumn/CSE 546/Assignments/HW2/A3d.png')
    plt.close()

    # output plot d
    for var in sets:
        plt.plot(lambda_list, error_dict[var], label = var, marker = 'o')
    plt.xscale('log')
    plt.title('Train and Test Error per Regularization Value')
    plt.xlabel('Lambda')
    plt.ylabel('Error')
    plt.legend(
        loc="lower right", 
        bbox_to_anchor=(1.1, 0.2), 
        fancybox=True, 
        shadow=True, 
        ncol=1
    )
    plt.savefig('/Users/hlukas/Google Drive/Grad School/2021-2022/Autumn/CSE 546/Assignments/HW2/A3e.png')
    plt.close()

    # output part e
    print('The largest weight ({}) is {}'.format(
        weights_30[max_30_idx], 
        colnames[max_30_idx])
    )
    print('The smallest weight ({}) is {}'.format(
        weights_30[min_30_idx], 
        colnames[min_30_idx])
    )
    #raise NotImplementedError("Your Code Goes Here")
    print(weight_dict['agePct12t29'])


if __name__ == "__main__":
    main()


# # %%

# df_train, df_test = load_dataset("crime")

# train_targets, test_targets = df_train['ViolentCrimesPerPop'].values, df_test['ViolentCrimesPerPop'].values
# train_obs = df_train.drop(['ViolentCrimesPerPop'], axis = 1).values
# test_obs = df_train.drop(['ViolentCrimesPerPop'], axis = 1).values

# train(X = train_obs, y = train_targets, _lambda = .5)
# # %%

# y_demean = train_targets - train_targets.mean()
# lambda_max = 2 * np.abs((train_obs.T * y_demean).sum(axis = 1)).max()

# lambda_val = lambda_max
# w = np.zeros(train_obs.shape[1])

# colnames = df_train.columns.tolist()
# vars = ['agePct12t29','pctWSocSec','pctUrban','agePct65up','householdsize']
# vars_idx = [colnames.index(x) for x in vars]

# lambda_list = []
# nonzero_weight_list = []
# weight_dict = {var: [] for var in vars}

# while lambda_val > .01:
#     weights,bias = train(
#         X = train_obs, 
#         y = train_targets, 
#         _lambda = lambda_val,
#         start_weight=w
#     )   

#     lambda_list.append(lambda_val)
#     nonzero_weight_list.append((weights > 1e-10).sum())

#     for i,var in enumerate(vars):
#         weight_dict[var].append(w[vars_idx[i]])

#     w = weights
#     lambda_val = lambda_val / 2

# weights_30,bias_30 = train(
#     X = train_obs, 
#     y = train_targets, 
#     _lambda = lambda_val,
#     start_weight=w
# )   

# max_30_idx = weights_30.index(weights_30.max())
# min_30_idx = weights_30.index(weights_30.min())

# print('The largest weight ({}) is {}'.format(weights_30[max_30_idx], colnames[max_30_idx]))
# print('The smallest weight ({}) is {}'.format(weights_30[min_30_idx], colnames[min_30_idx]))


# %%
