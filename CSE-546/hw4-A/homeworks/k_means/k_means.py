# %%

import numpy as np

from utils import problem


@problem.tag("hw4-A")
def calculate_centers(
    data: np.ndarray, classifications: np.ndarray, num_centers: int
) -> np.ndarray:
    """
    Sub-routine of Lloyd's algorithm that calculates the centers given datapoints and their respective classifications/assignments.
    num_centers is additionally provided for speed-up purposes.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        classifications (np.ndarray): Array of shape (n,) full of integers in range {0, 1, ...,  num_centers - 1}.
            Data point at index i is assigned to classifications[i].
        num_centers (int): Number of centers for reference.
            Might be usefull for pre-allocating numpy array (Faster that appending to list).

    Returns:
        np.ndarray: Array of shape (num_centers, d) containing new centers.
    """
    #raise NotImplementedError("Your Code Goes Here")

    n,d = data.shape
    new_centers = np.zeros((num_centers,d))
    

    for i,class_val in enumerate(range(num_centers)):
        indices = classifications == class_val
        center = np.mean(data[indices], 0)
        new_centers[i] += center

    return(new_centers)



@problem.tag("hw4-A")
def cluster_data(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Sub-routine of Lloyd's algorithm that clusters datapoints to centers given datapoints and centers.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        centers (np.ndarray): Array of shape (k, d). Each row is a center to which a datapoint can be clustered.

    Returns:
        np.ndarray: Array of integers of shape (n,), with each entry being in range {0, 1, 2, ..., k - 1}.
            Entry j at index i should mean that j^th center is the closest to data[i] datapoint.
    """
    #raise NotImplementedError("Your Code Goes Here")

    k,d = centers.shape
    n = data.shape[0]

    data_big = np.repeat(data, repeats = k, axis = 0)
    centers_big = np.vstack([centers]*n)

    diff = np.sum((data_big - centers_big) ** 2, 1)

    classifications = np.zeros(n)
    
    for i in range(n):
        classifications[i] = diff[i*k:(i+1)*k,].argmin()

    return(classifications)


@problem.tag("hw4-A")
def calculate_error(data: np.ndarray, centers: np.ndarray) -> float:
    """Calculates error/objective function on a provided dataset, with trained centers.

    Args:
        data (np.ndarray): Array of shape (n, d). Dataset to evaluate centers on.
        centers (np.ndarray): Array of shape (k, d). Each row is a center to which a datapoint can be clustered.
            These should be trained on training dataset.

    Returns:
        float: Single value representing mean objective function of centers on a provided dataset.
    """
    #raise NotImplementedError("Your Code Goes Here")
    k,d = centers.shape
    n = data.shape[0]
    classifications = cluster_data(data, centers)

    error = 0.

    for class_val in range(k):
        indices = classifications == class_val
        error += np.sum(np.sum((data[indices] - centers[class_val]) ** 2, 1) ** .5)

    return(error / n)

@problem.tag("hw4-A")
def lloyd_algorithm(
    data: np.ndarray, num_centers: int, epsilon: float = 10e-3
) -> np.ndarray:
    """Main part of Lloyd's Algorithm.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        num_centers (int): Number of centers to train/cluster around.
        epsilon (float, optional): Epsilon for stopping condition.
            Training should stop when max(abs(centers - previous_centers)) is smaller or equal to epsilon.
            Defaults to 10e-3.

    Returns:
        np.ndarray: Array of shape (num_centers, d) containing trained centers.

    Note:
        - For initializing centers please use the first `num_centers` data points.
    """
    #raise NotImplementedError("Your Code Goes Here")

    n,d = data.shape
    centers = data[:num_centers]
    previous_centers = np.ones((num_centers, d))

    while np.max(np.abs(centers - previous_centers)) > epsilon:
        previous_centers = centers.copy()
        classification = cluster_data(data, centers)
        centers = calculate_centers(data, classification, num_centers)

    return(centers)


# %%

# X = np.array(
#             [[1.0, 0.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0], [0.0, 0.5, 0.5],]
#         )
# centers = np.array([[0.5, 0.0, 0.5], [0.25, 0.5, 0.25],])

# k,d = centers.shape
# n = X.shape[0]
# classifications = cluster_data(X, centers)

# error = 0.

# for class_val in range(k):
#     indices = classifications == class_val
#     error += np.sum(np.sum((X[indices] - centers[class_val]) ** 2, 1) ** .5)

# print(error / 4)
# %%
