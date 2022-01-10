# %%
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import load_dataset, problem


@problem.tag("hw4-A")
def reconstruct_demean(uk: np.ndarray, demean_data: np.ndarray) -> np.ndarray:
    """Given a demeaned data, create a recontruction using eigenvectors provided by `uk`.

    Args:
        uk (np.ndarray): First k eigenvectors. Shape (d, k).
        demean_vec (np.ndarray): Demeaned data (centered at 0). Shape (n, d)

    Returns:
        np.ndarray: Array of shape (n, d).
            Each row should correspond to row in demean_data,
            but first compressed and then reconstructed using uk eigenvectors.
    """
    #raise NotImplementedError("Your Code Goes Here")
    return(demean_data @ uk @ uk.T)


@problem.tag("hw4-A")
def reconstruction_error(uk: np.ndarray, demean_data: np.ndarray) -> float:
    """Given a demeaned data and some eigenvectors calculate the squared L-2 error that recontruction will incur.

    Args:
        uk (np.ndarray): First k eigenvectors. Shape (d, k).
        demean_data (np.ndarray): Demeaned data (centered at 0). Shape (n, d)

    Returns:
        float: Squared L-2 error on reconstructed data.
    """
    #raise NotImplementedError("Your Code Goes Here")
    n,d = demean_data.shape
    return(np.sum((reconstruct_demean(uk, demean_data) - demean_data) ** 2) / n)


@problem.tag("hw4-A")
def calculate_eigen(demean_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given demeaned data calculate eigenvalues and eigenvectors of it.

    Args:
        demean_data (np.ndarray): Demeaned data (centered at 0). Shape (n, d)

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of two numpy arrays representing:
            1. Eigenvalues array with shape (d,)
            2. Matrix with eigenvectors as columns with shape (d, d)
    """
    # raise NotImplementedError("Your Code Goes Here")
    return(np.linalg.eig(demean_data.T @ demean_data / len(demean_data)))


@problem.tag("hw4-A", start_line=2)
def main():
    """
    Main function of PCA problem. It should load data, calculate eigenvalues/-vectors,
    and then answer all questions from problem statement.

    Part A:
        - Report 1st, 2nd, 10th, 30th and 50th largest eigenvalues
        - Report sum of eigenvalues

    Part C:
        - Plot reconstruction error as a function of k (# of eigenvectors used)
            Use k from 1 to 101.
            Plot should have two lines, one for train, one for test.
        - Plot ratio of sum of eigenvalues remaining after k^th eigenvalue with respect to whole sum of eigenvalues.
            Use k from 1 to 101.

    Part D:
        - Visualize 10 first eigenvectors as 28x28 grayscale images.

    Part E:
        - For each of digits 2, 6, 7 plot original image, and images reconstruced from PCA with
            k values of 5, 15, 40, 100.
    """
    (x_tr, y_tr), (x_test, _) = load_dataset("mnist")

    #raise NotImplementedError("Your Code Goes Here")

    demean_data_train = x_tr - np.mean(x_tr, 0)
    demean_data_test = x_test - np.mean(x_test, 0)

    eigs,vecs = calculate_eigen(demean_data_train)
    total_eigs = np.sum(eigs)

    print('PART A ------------------------------------------')
    for rank_val in [1,2,10,30,50]:
        print('Eigenvalue {}: {}'.format(rank_val, eigs[rank_val-1]))

    print('Sum of Eigenvalues: {}'.format(total_eigs))
    print('-------------------------------------------------')

    rec_error_train = []
    rec_error_test = []
    percentage = []
    for i in tqdm(range(1,102)):
        percentage.append(1-np.sum(eigs[:i])/total_eigs)

        rec_error_train.append(reconstruction_error(vecs[:, :i], demean_data_train))
        rec_error_test.append(reconstruction_error(vecs[:, :i], demean_data_test))

    plt.plot(list(range(1,102)), rec_error_train, 'o-b', label = 'Train')
    plt.plot(list(range(1,102)), rec_error_test, 'o-r', label = 'Test')
    plt.xlabel('Principal Components')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Error for Different Principal Components')
    plt.legend()
    plt.savefig('/Users/hlukas/Google Drive/Grad School/2021-2022/Autumn/CSE 546/Assignments/HW4/A5ca.png')
    plt.close()

    plt.plot(list(range(1,102)), percentage)
    plt.xlabel('Principal Components')
    plt.ylabel('Percentage of Total Eigenvalues')
    plt.title('Amount of Total Eigenvalues Accounted For')
    plt.savefig('/Users/hlukas/Google Drive/Grad School/2021-2022/Autumn/CSE 546/Assignments/HW4/A5cb.png')
    plt.close()

    f, axarr = plt.subplots(2,5)
    for i in range(10):
        axarr[int(i>4),i%5].imshow(vecs.T[i].reshape((28,28)), cmap='Greys')
        axarr[int(i>4),i%5].title.set_text('{}'.format(i+1))
    f.tight_layout()
    f.suptitle('Visualized Eigenvectors for MNIST Dataset')
    f.savefig('/Users/hlukas/Google Drive/Grad School/2021-2022/Autumn/CSE 546/Assignments/HW4/A5d.png')
    plt.close('all')

    f, axarr = plt.subplots(3,5)
    for i,img in enumerate([25,13,15]):
        axarr[i,0].imshow(x_tr[img].reshape((28,28)), cmap='Greys')
        axarr[i,0].title.set_text('Original Image')
        axarr[i,1].imshow(reconstruct_demean(vecs[:,:5], demean_data_train[img]).reshape((28,28)), cmap='Greys')
        axarr[i,1].title.set_text('k=5')
        axarr[i,2].imshow(reconstruct_demean(vecs[:,:15], demean_data_train[img]).reshape((28,28)), cmap='Greys')
        axarr[i,2].title.set_text('k=15')
        axarr[i,3].imshow(reconstruct_demean(vecs[:,:40], demean_data_train[img]).reshape((28,28)), cmap='Greys')
        axarr[i,3].title.set_text('k=40')
        axarr[i,4].imshow(reconstruct_demean(vecs[:,:100], demean_data_train[img]).reshape((28,28)), cmap='Greys')
        axarr[i,4].title.set_text('k=100')

    f.tight_layout()
    f.suptitle('Images from MNIST Dataset and Reconstructions')
    f.savefig('/Users/hlukas/Google Drive/Grad School/2021-2022/Autumn/CSE 546/Assignments/HW4/A5e.png')
    plt.close('all')

    f, axarr = plt.subplots(3,4)
    for i,img in enumerate([25,13,15]):
        axarr[i,0].imshow(x_tr[img].reshape((28,28)), cmap='Greys')
        axarr[i,0].title.set_text('Original Image')
        axarr[i,1].imshow(reconstruct_demean(vecs[:,:32], demean_data_train[img]).reshape((28,28)), cmap='Greys')
        axarr[i,1].title.set_text('k=32')
        axarr[i,2].imshow(reconstruct_demean(vecs[:,:64], demean_data_train[img]).reshape((28,28)), cmap='Greys')
        axarr[i,2].title.set_text('k=64')
        axarr[i,3].imshow(reconstruct_demean(vecs[:,:128], demean_data_train[img]).reshape((28,28)), cmap='Greys')
        axarr[i,3].title.set_text('k=128')

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    f.tight_layout()
    #f.suptitle('Images from MNIST Dataset and Reconstructions')
    f.savefig('/Users/hlukas/Google Drive/Grad School/2021-2022/Autumn/CSE 546/Assignments/HW4/A6d.png')
    plt.close('all')

# %%
if __name__ == "__main__":
    main()

# %%

# (x_tr, y_tr), (x_test, _) = load_dataset("mnist")

# #raise NotImplementedError("Your Code Goes Here")

# demean_data = x_tr - np.mean(x_tr, 0)

# eigs,vecs = calculate_eigen(demean_data)
# %%
