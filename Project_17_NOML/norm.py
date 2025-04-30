import argparse

import numpy as np
import time
import matplotlib.pyplot as plt
from torch.distributions.constraints import real_vector

from gradient_descent import gradient_descent
from bfgs import bfgs
from lbfgs import lbfgs
from function import *

# Function that allows to create random matrices of size A \in R^{NxM}
# This function returns A.T @ A | \lambda_{max}=100 to allow comparisons on plots
def rand(n, m, max_eigenvalue=100):
    # Create a random n × m matrix
    A = np.random.randn(n, m)

    # Compute A.T @ A (which is m × m and symmetric positive semi-definite)
    ATA = A.T @ A

    # Compute max eigenvalue of A.T @ A
    eigvals = np.linalg.eigvalsh(ATA)  # eigvalsh is faster for symmetric matrices
    max_eigval = np.max(eigvals)

    # Scale A so that the max eigenvalue of A.T @ A becomes max_eigenvalue
    scaling_factor = np.sqrt(max_eigenvalue / max_eigval)
    A_scaled = A * scaling_factor

    return A_scaled.T @ A_scaled


# Random start vector generator
def random_start_vector(n, min_norm=1):
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    v = v * np.random.uniform(min_norm, 2 * min_norm)
    return v # Return a vector such that the norm is min_norm < ||v|| < 2 * min_norm

# Utility function to pad lists of trials to the longest to be able to trace the median on the plot
def pad_to_longest(data_lists):
    max_len = max(len(lst) for lst in data_lists)
    return np.array([
        np.pad(lst, (0, max_len - len(lst)), mode='edge')
        if len(lst) > 0 else np.zeros(max_len)
        for lst in data_lists
    ])


def main():
    # The the args from input
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', nargs='+', type=int, default=[250, 1000, 2500],
                        help='Matrices sizes')
    parser.add_argument('--seed', type=int, default=18042001, help='Random seed')
    parser.add_argument('--max_eig', type=int, default=100, help='Max eigenvalue')
    parser.add_argument('--num_trials', type=int, default=5, help='Num of trials')
    parser.add_argument('--iters', type=int, default=5, help='Max iterations')
    args = parser.parse_args()

    np.random.seed(args.seed)
    sizes = args.sizes
    max_eig = args.max_eig
    num_trials = args.num_trials
    iters = args.iters

    for size1 in sizes:
      for size2 in sizes:
        print(f"\nProcessing matrix of size {size1}x{size2} over {num_trials} trials...\n")

        # List to store gradient errors of all trials and functions values of all trials for gd
        all_err_gd, all_fx_gd = [], []
        # List to store gradient errors of all trials and functions values of all trials for bfgs
        all_err_bfgs, all_fx_bfgs = [], []
        # List to store gradient errors of all trials and functions values of all trials for lbfgs
        all_err_lbfgs, all_fx_lbfgs = [], []

        # Generate random matrix \in R^{NxM} and return Q = A.T @ A | max_eigenvalue = max_eig
        # Random Q matrix (A^T @ A) fixed among all the trials of the different algorithms
        Q = rand(size1, size2, max_eig)

        for trial in range(num_trials):
            # Generate random starting vector with norm greater than 1
            v0 = random_start_vector(size2)

            # Gradient Descent
            start_time = time.time()
            v_gd, err_gd, fx_gd, its = gradient_descent(Q, v0)
            elapsed_time_gd = time.time() - start_time
            print(f"[Trial {trial + 1}] Gradient Descent: Time = {elapsed_time_gd:.4f}s, Iterations = {its}")
            all_err_gd.append(err_gd)
            all_fx_gd.append(fx_gd)

            # BFGS
            start_time = time.time()
            v_bfgs, err_bfgs, fx_bfgs, its = bfgs(Q, v0)
            elapsed_time_bfgs = time.time() - start_time
            print(f"[Trial {trial + 1}] BFGS: Time = {elapsed_time_bfgs:.4f}s, Iterations = {its}")
            all_err_bfgs.append(err_bfgs)
            all_fx_bfgs.append(fx_bfgs)

            # L-BFGS
            start_time = time.time()
            v_lbfgs, err_lbfgs, fx_lbfgs, its = lbfgs(Q, v0, m=10)
            elapsed_time_lbfgs = time.time() - start_time
            print(f"[Trial {trial + 1}] L-BFGS: Time = {elapsed_time_lbfgs:.4f}s, Iterations = {its}")
            all_err_lbfgs.append(err_lbfgs)
            all_fx_lbfgs.append(fx_lbfgs)

        # Calculate the true minimum with np.eigenvalues on Q = (A^T A)
        real_max = np.max(np.linalg.eigvals(Q))
        if isinstance(real_max, complex):
            real_max = real_max.real

        # Replace function values with gap from the true minimum
        all_fx_gd = [np.array(fx) + real_max for fx in all_fx_gd]
        all_fx_bfgs = [np.array(fx) + real_max for fx in all_fx_bfgs]
        all_fx_lbfgs = [np.array(fx) + real_max for fx in all_fx_lbfgs]

        # Uniform length for averaging
        all_err_gd = pad_to_longest(all_err_gd)
        all_fx_gd = pad_to_longest(all_fx_gd)
        all_err_bfgs = pad_to_longest(all_err_bfgs)
        all_fx_bfgs = pad_to_longest(all_fx_bfgs)
        all_err_lbfgs = pad_to_longest(all_err_lbfgs)
        all_fx_lbfgs = pad_to_longest(all_fx_lbfgs)

        # Plot Errors
        plt.figure(figsize=(10, 5))
        for e in all_err_gd:
            plt.plot(e, color="blue", alpha=0.4)
        for e in all_err_bfgs:
            plt.plot(e, color="red", alpha=0.4)
        for e in all_err_lbfgs:
            plt.plot(e, color="green", alpha=0.4)

        # Plot the medians of gradient norm of all algorithms
        plt.plot(np.median(all_err_gd, axis=0), label="GD (median)", color="blue", linewidth=2)
        plt.plot(np.median(all_err_bfgs, axis=0), label="BFGS (median)", color="red", linewidth=2)
        plt.plot(np.median(all_err_lbfgs, axis=0), label="L-BFGS (median)", color="green", linewidth=2)

        plt.yscale("log")
        plt.xlabel("Iterations")
        plt.ylabel("Gradient error")
        plt.title(f"Gradient error for Matrix Size {size1}x{size2} | max_eigenvalue = {max_eig}")
        plt.legend()
        plt.grid()
        plt.savefig(f"gradient_error_{size1}x{size2}_eig{max_eig}.png")

        # Plot the gap error between function value and true min => (f_k - np.max(np.eigenvals(Q))
        plt.figure(figsize=(10, 5))
        for f in all_fx_gd:
            plt.plot(f, color="blue", alpha=0.4)
        for f in all_fx_bfgs:
            plt.plot(f, color="red", alpha=0.4)
        for f in all_fx_lbfgs:
            plt.plot(f, color="green", alpha=0.4)

        # Plot the median gap error between function value and true min => (f_k - np.max(np.eigenvals(Q)) of all algorithms
        plt.plot(np.median(all_fx_gd, axis=0), label="GD (median)", color="blue", linewidth=2)
        plt.plot(np.median(all_fx_bfgs, axis=0), label="BFGS (median)", color="red", linewidth=2)
        plt.plot(np.median(all_fx_lbfgs, axis=0), label="L-BFGS (median)", color="green", linewidth=2)

        plt.yscale("log")
        plt.xlabel("Iterations")
        plt.ylabel("f(x) error from minimum")
        plt.title(f"Gap error for Matrix Size {size1}x{size2} | max_eigenvalue = {max_eig}")
        plt.legend()
        plt.grid()
        plt.savefig(f"function_error_{size1}x{size2}_eig{max_eig}.png")


if __name__ == "__main__":
    main()
