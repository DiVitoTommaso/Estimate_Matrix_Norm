import numpy as np
from function import *

# BFGS Optimization Algorithm to minimize function(x) = 0.5 * xᵀQx
# where Q is symmetric positive semi-definite
def bfgs(Q, x0, max_iter=300):
    x = x0.copy()          # Initial point (make a copy to avoid mutating the input)
    d = len(x)             # Dimensionality of the input
    H = np.eye(d)          # Initial inverse Hessian approximation as identity matrix

    fx = []                # List to track function values over iterations
    err = []               # List to track gradient norm (error) over iterations
    f_prev = function(Q, x)  # Evaluate the initial function value

    for i in range(max_iter):
        fx.append(function(Q, x))         # Store current function value
        g = gradient(Q, x)                # Compute gradient at current point
        grad_norm = np.linalg.norm(g)     # Compute L2 norm of the gradient
        err.append(grad_norm)             # Store gradient norm

        # Check convergence: if gradient norm is sufficiently small, stop
        if grad_norm <= 1e-6:
            break

        # Compute descent direction using inverse Hessian approximation
        p = -H @ g

        # Find optimal step size in direction p using exact line search
        alpha = exact_line_search(x, p, Q)

        # Step vector: s = αp
        s = p * alpha
        x_new = x + s                    # Update the solution estimate

        # Compute gradient at new point and define y = g_{k+1} - g_k
        g_new = gradient(Q, x_new)
        y = g_new - g

        # Reshape s and y to column vectors for matrix math
        sk = s.reshape(-1, 1)
        yk = y.reshape(-1, 1)

        # Update inverse Hessian approximation using BFGS formula
        term1 = ((sk.T @ yk + yk.T @ H @ yk) * (sk @ sk.T)) / ((yk.T @ sk) ** 2)
        term2 = (H @ yk @ sk.T + sk @ yk.T @ H) / (sk.T @ yk)
        H = H + term1 - term2
        x = x_new

        # Decrease check on function value
        f_curr = function(Q, x_new)
        f_check(f_curr, f_prev, 'BFGS')
        f_prev = f_curr

    return x, err, fx, i+1               # Return final point, gradient error list, function values list, and iteration count
