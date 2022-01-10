# %%

import numpy as np
import matplotlib.pyplot as plt

def main():

    n = 20000
    d = 10000

    # define the values of x_i and y_i

    X = ((np.mod(np.arange(n), d) + 1) ** .5).reshape(n,1) * np.vstack([np.eye(d), np.eye(d)])
    y = np.random.normal(0,1,n)

    # compute the estimator

    inv = np.diag(1. / (2.*(np.arange(1,d+1))))
    beta = inv @ X.T @ y

    # compute the CIs
    ci_plus = ((1. / (np.arange(1,d+1))) * np.log(2 / .05)) ** .5
    ci_minus = -ci_plus

    plt.scatter(np.arange(d), beta, facecolors='none', edgecolors='b', alpha = .15, s = 1.5)
    plt.plot(np.arange(d), ci_plus, 'r--')
    plt.plot(np.arange(d), ci_minus, 'r--')
    plt.title('Values of Beta')
    plt.xlabel('Beta Index')
    plt.ylabel('Weight')
    plt.savefig('/Users/hlukas/Google Drive/Grad School/2021-2022/Autumn/CSE 546/Assignments/HW2/B4c.png')

    outside = np.any([(beta > ci_plus), (beta < ci_minus)], axis = 0).sum()
    print('{} coefficients fell outside the confidence interval ({}%)'.format(outside, outside * 100 / d))

if __name__ == '__main__':
    main()