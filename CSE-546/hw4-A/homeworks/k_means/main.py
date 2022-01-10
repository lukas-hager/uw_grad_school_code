# %% 
if __name__ == "__main__":
    from k_means import calculate_error, lloyd_algorithm  # type: ignore
else:
    from .k_means import lloyd_algorithm, calculate_error

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem

# %%
@problem.tag("hw4-A", start_line=1)
def main():
    """Main function of k-means problem

    You should:
        a. Run Lloyd's Algorithm for k=10, and report 10 centers returned.
        b. For ks: 2, 4, 8, 16, 32, 64 run Lloyd's Algorithm,
            and report objective function value on both training set and test set.
            (All one plot, 2 lines)

    NOTE: This code takes a while to run. For debugging purposes you might want to change:
        x_train to x_train[:10000]. CHANGE IT BACK before submission.
    """
    (x_train, _), (x_test, _) = load_dataset("mnist")
    #raise NotImplementedError("Your Code Goes Here")

    debug = False

    if debug == True:
         x_train = x_train[:10000]
    
    print('Running 10 clusters')
    centers_10 = lloyd_algorithm(x_train,10)

    f, axarr = plt.subplots(2,5)
    for i in range(10):
        axarr[int(i>4),i%5].imshow(centers_10[i].reshape((28,28)), cmap='Greys')
    f.tight_layout()
    f.suptitle('Visualized Clusters for MNIST Dataset')
    f.savefig('/Users/hlukas/Google Drive/Grad School/2021-2022/Autumn/CSE 546/Assignments/HW4/A4b.png')
    plt.close('all')

    # k_vals = [2,4,8,16,32,64]
    # train_error = np.zeros(len(k_vals))
    # test_error = np.zeros(len(k_vals))
    # for i,k in enumerate([2,4,8,16,32,64]):
    #     print('Running {} clusters'.format(k))
    #     centers = lloyd_algorithm(x_train,k)
    #     train_error[i] += calculate_error(x_train, centers)
    #     test_error[i] += calculate_error(x_test, centers)
    #     print(calculate_error(x_train, centers))
    #     print(calculate_error(x_test, centers))

    # plt.plot(k_vals, train_error, 'o-b', label = 'Train')
    # plt.plot(k_vals, test_error, 'o-r', label = 'Test')
    # plt.legend()
    # plt.xlabel('k')
    # plt.ylabel('Error')
    # plt.savefig('/Users/hlukas/Google Drive/Grad School/2021-2022/Autumn/CSE 546/Assignments/HW4/A4c.png')


if __name__ == "__main__":
    main()

# %%
#(x_train, _), (x_test, _) = load_dataset("mnist")

# f, axarr = plt.subplots(2,5)
# for i in range(10):
#     axarr[i%2,i%5].imshow(centers_10[i].reshape((28,28)))