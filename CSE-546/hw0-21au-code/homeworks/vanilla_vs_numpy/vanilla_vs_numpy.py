# %%

# Import numpy <- You will see this line a lot
# Import time for testing function
import time

# Import typing support
from typing import List

import matplotlib.pyplot as plt
import numpy as np

# And our custom utilities library
from utils import problem

Vector = List[float]
Matrix = List[Vector]


# %%
def vanilla_matmul(A: Matrix, x: Vector) -> Vector:
    """Performs a matrix multiplications Ax on vanilla python lists.

    Args:
        A (Matrix): a (n, d) matrix.
        x (Vector): a (d,) vector.

    Returns:
        Vector: a resulting (n,) vector.

    Note:
        In this problem specifically d = n, since A and B are square matrices.
    """
    result: Vector = []
    for a in A:
        result_a = 0.0
        for a_i, x_i in zip(a, x):
            result_a += a_i * x_i
        result.append(result_a)
    return result


# %%
def vanilla_transpose(A: Matrix) -> Matrix:
    """Performs a matrix transpose

    Args:
        A (Matrix): a (n, d) matrix.

    Returns:
        Matrix: a resulting (d, n) matrix.
    """
    result: Matrix = [[] for _ in A[0]]  # Create list of d lists
    for a in A:
        for jdx, value in enumerate(a):
            result[jdx].append(value)
    return result


# %%
@problem.tag("hw0-A")
def vanilla_solution(x: Vector, y: Vector, A: Matrix, B: Matrix) -> Vector:
    """Calculates gradient of f(x, y) with respect to x using vanilla python lists.
    Where $$f(x, y) = x^T A x + y^T B x + c$$

    Args:
        x (Vector): a (n,) vector.
        y (Vector): a (n,) vector.
        A (Matrix): a (n, n) matrix.
        B (Matrix): a (n, n) matrix.

    Returns:
        Vector: a resulting (n,) vector.

    Note:
        - We provided you with `vanilla_transpose` and `vanilla_matmul` functions which you should use.
        - In this context (and documentation of two functions above) vector means list of floats,
            and matrix means list of lists of floats
    """

    grad_x = []

    # implement summation by matrix multiplying and selecting the relevant element for each 
    # part of the gradient

    for i in range(len(x)):
        term_1 = vanilla_matmul(A,x)[i] 
        term_2 = vanilla_matmul(vanilla_transpose(A),x)[i] 
        term_3 = vanilla_matmul(vanilla_transpose(B),y)[i]
        grad_x.append(term_1 + term_2 + term_3)

    return(grad_x)

# TEST

# vanilla_solution([1,1], [1,1], [[0,1],[2,3]], [[0,1],[2,3]])
# %%
@problem.tag("hw0-A")
def numpy_solution(
    x: np.ndarray, y: np.ndarray, A: np.ndarray, B: np.ndarray
) -> np.ndarray:
    """Calculates gradient of f(x, y) with respect to x using numpy arrays.
    Where $$f(x, y) = x^T A x + y^T B x + c$$

    Args:
        x (np.ndarray): a (n,) numpy array.
        y (np.ndarray): a (n,) numpy array.
        A (np.ndarray): a (n, n) numpy array.
        B (np.ndarray): a (n, n) numpy array.

    Returns:
        np.ndarray: a resulting (n, ) numpy array.

    Note:
        - Make use of numpy docs: https://numpy.org/doc/
            You will use this link a lot throughout quarter, so it might be a good idea to bookmark it!
    """
    term_1 = np.matmul(A,x)
    term_2 = np.matmul(np.transpose(A), x)
    term_3 = np.matmul(np.transpose(B), y)
    grad_x = term_1 + term_2 + term_3
    return(grad_x)

# TEST

# numpy_solution(
#     np.ones(2).reshape((2,1)), 
#     np.ones(2).reshape((2,1)),
#     np.arange(4).reshape((2,2)), 
#     np.arange(4).reshape((2,2))
# )
# %%
def main():
    """
    Before running this file run `inv test` to make sure that your numpy as vanilla solutions are correct.
    """
    RNG = np.random.RandomState(seed=446)

    ns = [20, 200, 500, 1000]
    vanilla_times = []
    numpy_times = []

    for n in ns:
        # Generate some data
        x = RNG.randn(n)
        y = RNG.randn(n)
        A = RNG.randn(n, n)
        B = RNG.randn(n, n)
        # And their vanilla List equivalents
        x_list = x.tolist()
        y_list = y.tolist()
        A_list = A.tolist()
        B_list = B.tolist()

        start = time.time_ns()
        vanilla_result = vanilla_solution(x_list, y_list, A_list, B_list,)
        vanilla_time = time.time_ns() - start
        start = time.time_ns()
        numpy_result = numpy_solution(x, y, A, B)
        numpy_time = time.time_ns() - start

        np.testing.assert_almost_equal(vanilla_result, numpy_result)

        print(f"Time for vanilla implementation: {vanilla_time / 1e6}ms")
        print(f"Time for numpy implementation: {numpy_time / 1e6}ms")

        vanilla_times.append(vanilla_time)
        numpy_times.append(numpy_time)

    plt.plot(ns, vanilla_times, label="Vanilla")
    plt.plot(ns, numpy_times, label="Numpy")
    plt.xlabel("n")
    plt.ylabel("Time (ns)")
    plt.legend()
    plt.savefig('/Users/hlukas/Google Drive/Grad School/2021-2022/Autumn/CSE 546/Assignments/HW0/A09.png')

if __name__ == "__main__":
    main()