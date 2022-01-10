## Problem Set 1
#### Lukas Hager
#### ECON 525
___
All the code produced here exists in the file `problem_set_1.py`. This `README.md` file will serve as the answers to the problems, which is rendered more nicely in `README.html`.
___
1. To get the machine epsilon, we iterate over a grid on $[0,1]$ and find the smallest value by iteratively dividing by 2: 
    ```python
    import numpy as np

    def get_mach_eps(init_val):
        eps_diff = False
        eps = init_val

        while not eps_diff:
            if 1 + eps > 1-eps:
                eps /= 2
            else:
                eps_diff = True

        return(eps)

    machine_epsilons = []
    for value in np.arange(.01,1,.01):
        machine_epsilons.append(get_mach_eps(value))

    print('1. The machine epsilon is {}'.format(min(machine_epsilons)))
    ```
    The machine epsilon winds up being $2.831068712794149\times 10^{-17}$

2. The code to produce the L-U decomposition is below:
    ```python
    # i know this won't work in all cases, but it works for this one (doesn't prevent division by zero)

    def get_lu_diag(mat):

        mat_copy = mat.copy()

        mat_dim = mat.shape
        iden = np.eye(mat_dim[0])

        for col in range(mat_dim[1]):
            eliminator = mat_copy[col, col]
            for row in range(col+1, mat_dim[0]):
                val_to_eliminate = mat_copy[row, col]
                elim_amount = val_to_eliminate / eliminator
                mat_copy[row,] -= mat_copy[col,] * elim_amount
                iden[row,col:] += iden[col,col:] * elim_amount

        return((iden,mat_copy))
    ```
    To then backsubstitute, use the code below:
    ```python 
    def backsub(mat, vec):

        n = mat.shape[0]
        y = np.zeros(n)

        # figure out whether upper or lower triangular
        is_upper = int(sum(mat[0, 1:]) > 0)

        for row in range((n-1)*is_upper, n*(1-is_upper) -1 * is_upper, (-1)**is_upper):
            if row == n * is_upper:
                y[row] += vec[row] / mat[row,row]
            else:
                new_val = (vec[row] - mat[row] @ y) / mat[row,row]
                y[row] += new_val

        return(y)
    ```
    So then to solve the problem, we do
    ```python
    mat = np.array([[54., 14., -11., 2.],
    [14., 50., -4., 29.],
    [-11., -4., 55., 22.],
    [2., 29., 22., 95.]])

    lower,upper = get_lu_diag(mat)

    y_val = backsub(lower, np.ones(mat.shape[0]))
    x_val = backsub(upper, y_val)   
    ```
    The Gauss-Jacobi iteration is below:
    ```python
    def gauss_jacobi(mat, vec, answer):

        mat_shape = mat.shape[0]
        guess = np.ones(mat_shape)
        iter = 0

        while np.max(np.abs(guess - answer)) > .000001:
            iter += 1
            new_guess = np.zeros(mat_shape)
            for row in range(mat_shape):
                new_guess[row] = (1 / mat[row,row]) * (vec[row] - (np.delete(mat[row,], row) @ np.delete(guess, row)))
            guess = new_guess

        return(iter)
    ```
    The Gauss-Seidel iteration is below:
    ```python
    def gauss_seidel(mat, vec, answer):

        mat_shape = mat.shape[0]
        guess = np.ones(mat_shape)
        iter = 0

        while np.max(np.abs(guess - answer)) > .000001:
            iter += 1
            for row in range(mat_shape):
                guess[row] = (1 / mat[row,row]) * (vec[row] - (np.delete(mat[row,], row) @ np.delete(guess, row)))

        return(iter)

    
    ```
    So to get the iterations, we simply run
    ```python
    gj_iters = gauss_jacobi(mat, np.ones(4), x_val)
    gs_iters = gauss_seidel(mat, np.ones(4), x_val)
    ```
    We see that Gauss-Jacobi requires **22 iterations** while Gauss-Seidel requires only **13 iterations**.

3. The code to use all these algorithms is produced below:
    ```python
    def f1(x):
        sgns = np.sign(x)
        return(np.abs(x[0])**.2 * sgns[0] + np.abs(x[1])**.2 * sgns[1] - 2)

    def f2(x):
        sgns = np.sign(x)
        return(np.abs(x[0])**.1 * sgns[0] + np.abs(x[1])**.4 * sgns[1] - 2)

    def f1_1(x):
        return(.2*np.abs(x[0])**(-.8))

    def f2_2(x):
        sgns = np.sign(x)
        return(.4*np.abs(x[1])**(-.6) * sgns[1])

    def f1_2(x):
        return(.2*np.abs(x[1])**(-.8))

    def f2_1(x):
        sgns = np.sign(x)
        return(.1*np.abs(x[1])**(-.9) * sgns[1])

    def gauss_jacobi_nl(fun_list, guess):

        iter = 0
        converged = False

        while converged == False:
            iter += 1
            new_guess = np.zeros(2)
            for row in range(2):
                new_guess[row] = guess[row] - fun_list[row](guess) / fun_list[row+2](guess)
            if np.max(np.abs(guess - new_guess)) <= .0001:
                converged = True
            else:
                guess = new_guess
            
        return(new_guess,iter)

    def gauss_seidel_nl(fun_list, guess):

        iter = 0
        converged = False

        while converged == False:
            iter += 1
            lag_guess = guess.copy()
            for row in range(2):
                val = guess[row] - fun_list[row](guess) / fun_list[row+2](guess)
                guess[row] = val
            if np.max(np.abs(guess - lag_guess)) <= .0001:
                converged = True
            
        return(guess,iter)

    def get_jacobian(x):
        J = np.array([f1_1(x), f1_2(x), f2_1(x), f2_2(x)]).reshape((2,2))
        return(np.linalg.inv(J))

    def newton(guess):

        iter = 0
        crit = False
        while crit == False:
            iter += 1
            new_guess = guess - get_jacobian(guess) @ np.array([f1(guess), f2(guess)])
            if np.max(np.abs(guess - new_guess)) <= .0001:
                crit = True
            else:
                guess = new_guess
        
        return(new_guess, iter)

    def broyden(guess):

        iter = 0
        crit = False
        A = np.eye(2)        

        while crit == False:
            iter += 1
            s = - np.linalg.inv(A) @ np.array([f1(guess), f2(guess)])
            new_guess = guess + s
            y = np.array([f1(new_guess), f2(new_guess)]) - np.array([f1(guess), f2(guess)])
            new_guess = guess - get_jacobian(guess) @ np.array([f1(guess), f2(guess)])
            new_A = (A + (y - A @ s) @ s.T) / (s.T @ s)
            if np.max(np.abs(guess - new_guess)) <= .0001:
                crit = True
            else:
                guess = new_guess
        
        return(new_guess, iter)
    ```
    To get the results, we run
    ```python
    print(gauss_jacobi_nl([f1,f2,f1_1,f2_2], np.array([3.,3.])))
    print(gauss_seidel_nl([f1,f2,f1_1,f2_2], np.array([3.,3.])))
    print(gauss_jacobi_nl([f1,f2,f1_1,f2_2], np.array([2.,2.])))
    print(gauss_seidel_nl([f1,f2,f1_1,f2_2], np.array([2.,2.])))
    print(gauss_jacobi_nl([f1,f2,f1_1,f2_2], np.array([1.5,1.5])))
    print(gauss_seidel_nl([f1,f2,f1_1,f2_2], np.array([1.5,1.5])))

    print(newton([2.,2.]))
    print(newton([3.,3.]))
    print(broyden([2.,2.]))
    print(broyden([3.,3.]))
    ```
    Our results differ from Judd, 1998 for a few reasons. First, I'm not sure that Judd computes the solution using Gauss-Jacobi or Gauss-Seidel; for these algorithms, my solution will converge with an initial guess of 1.5, but not for an initial guess of 2 or 3. The result contained in Judd is that using Newton's Method or Broyden's Method, the algorithm will not converge for an initial guess of 3, but this is due to the fact that *they assume the computer will return a complex number when asked to take the root of a negative number*. If we write the code to return the real root (using knowledge of the specific problem), as I have above, the solution will converge. 
    
    So to summarize:

    **Gauss-Jacobi, [3,3]**: No convergence

    **Gauss-Jacobi, [2,2]**: No convergence

    **Gauss-Jacobi, [1.5,1.5]**: Convergence

    **Gauss-Seidel, [3,3]**: No convergence
    
    **Gauss-Seidel, [2,2]**: No convergence
    
    **Gauss-Seidel, [1.5,1.5]**: Convergence

    **Newton, [3,3]**: Convergence
    
    **Newton, [2,2]**: Convergence

    **Broyden, [3,3]**: Convergence

    **Broyden, [2,2]**: Convergence