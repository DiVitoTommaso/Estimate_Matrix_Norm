import numpy as np

# Warning for function value increased. Debug function
def f_check(curr, prev, method):
    if curr > prev + 1e-9:
        print(f"[{method}] Warning: function increased! f_prev = {prev:.6e}, f_curr = {curr:.6e}")


# Compute function f(x)
def function(Q, x):
    xTQx = -(x.T @ Q @ x)  # x^T Q x
    xTx = x.T @ x  # x^T x
    return xTQx / xTx


# Compute the gradient of f(x)
def gradient(Q, x):
    grad = -2 * ((Q @ x) * (x.T @ x) - (x.T @ Q @ x) * x) / ((x.T @ x) ** 2)  # Apply quotient rule
    return grad

# Compute the line search with the formula calculated in the report
def exact_line_search(v, d, M):
    p = v.T @ v
    q = 2 * (v.T @ d)
    r = d.T @ d

    a = v.T @ M @ v
    b = 2 * (v.T @ M @ d)
    c = d.T @ M @ d

    A = -b * r + c * q
    B = 2 * c * p - 2 * a * r
    C = b * p - a * q

    discriminant = B ** 2 - 4 * A * C

    if discriminant < 0:
        raise ValueError("No real roots found for step size")

    sqrt_disc = np.sqrt(discriminant)
    alpha1 = (-B + sqrt_disc) / (2 * A)
    alpha2 = (-B - sqrt_disc) / (2 * A)

    f1 = function(M, v + alpha1 * d)
    f2 = function(M, v + alpha2 * d)

    # Pick the positive value if one is negative
    if alpha1 < 0:
        return alpha2
    if alpha2 < 0:
        return alpha1

    # If both are positive pick the one that minimize f(v+ad)
    alpha_opt = alpha1 if f1 < f2 else alpha2

    return alpha_opt