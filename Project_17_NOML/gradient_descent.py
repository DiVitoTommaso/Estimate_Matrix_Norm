import numpy as np

from function import *

# Gradient Descent
def gradient_descent(Q, v0, max_iter=300):
    v = v0.copy()
    err = []
    fx = []
    f_prev = function(Q, v)
    for i in range(max_iter):
        # Calculate the gradient
        grad = -gradient(Q, v)
        # Append current values
        fx.append(function(Q, v))
        err.append(np.linalg.norm(grad))
        # Do line search
        alpha = exact_line_search(v, grad, Q)

        # Stops if gradient norm becomes very small
        if np.linalg.norm(grad) < 1e-6:
            break

        # Update rule for gradient descent
        v = v + alpha * grad

        # Check that function value always decrease
        f_curr = function(Q, v)
        f_check(f_curr, f_prev, 'GD')
        f_prev = f_curr

    return v, err, fx, i+1 # return vector, gradient error list, function values list and iterations count