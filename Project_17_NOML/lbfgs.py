import numpy as np

from function import *
# 1:1 L-BFGS Algorithm implementation taken from wikipedia
def lbfgs(Q, v, m=10, max_iter=300):
    S = []  # List to store past steps
    Y = []  # List to store past gradient differences
    g = gradient(Q, v)  # Compute initial gradient

    fx = []
    err = []
    f_prev = function(Q, v)

    for k in range(max_iter):
        q = g.copy()  # Initialize q for two-loop recursion
        alpha = []
        rho = []

        # save current values of gradient norm and function value
        fx.append(function(Q, v))
        grad_norm = np.linalg.norm(g)
        err.append(grad_norm)

        # First loop (from newest to oldest)
        for i in reversed(range(len(S))):
            rho_i = 1.0 / (Y[i].T @ S[i])
            alpha_i = rho_i * (S[i].T @ q)
            q = q - alpha_i * Y[i]
            alpha.append(alpha_i)
            rho.append(rho_i)
        alpha = alpha[::-1]  # Reverse alpha for second loop
        rho = rho[::-1]

        if len(S) > 0:
            H0_scalar = (S[-1].T @ Y[-1]) / (Y[-1].T @ Y[-1])
        else:
            H0_scalar = 1.0
        # Initial Hessian approximation (scalar times identity)
        H0 = H0_scalar * np.eye(len(v))

        p = (H0 @ q)  # Initial search direction

        if np.linalg.norm(g) < 1e-6:
            break  # Stop if gradient norm is small enough

        # Second loop (from oldest to newest)
        for i in range(len(S)):
            beta = rho[i] * (Y[i].T @ p)
            p = p + S[i] * (alpha[i] - beta)

        p = -p
        # Compute optimal step size
        alpha_step = exact_line_search(v, p, Q)
        s = alpha_step * p
        v_new = v + s
        g_new = gradient(Q, v_new)
        y = g_new - g

        # Update memory (S, Y), If history is full discard the oldest element
        if len(S) == m:
            S.pop(0)
            Y.pop(0)
        S.append(s)
        Y.append(y)

        # Update current point and gradient
        v = v_new
        g = g_new

        # Check that function value always decrease
        f_curr = function(Q, v_new)
        f_check(f_curr, f_prev, 'L-BFGS')
        f_prev = f_curr

    return v, err, fx, k+1, # return vector, gradient error list, function values list and iterations count
