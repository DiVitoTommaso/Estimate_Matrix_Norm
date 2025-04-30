# Project 17

This project compares three optimization algorithms — **Gradient Descent**, **BFGS**, and **L-BFGS** — for estimating the matrix norm of randomly generated matrices.

## Project Structure

    .
    ├── norm.py     
    ├── function.py    
    ├── gradient_descent.py
    ├── bfgs.py  
    ├── lbfgs.py        
    ├── README.md     

## Requirements

Written in Python 3. Required libraries:

- numpy
- matplotlib

## Basic Execution

To run the default benchmark:

    python norm.py

This will output, for each algorithm:
- Execution time
- Number of iterations

It also generates two `.png` plots for each size NxM:
- Gradient norm error vs iterations (log scale)
- Function value gap from minimum vs iterations

Files will be named like:

    gradient_error_NxM_eigK.png
    function_error_NxM_eigK.png

## CLI Parameterized Execution

You can customize the test using command-line arguments:

    python main.py --size 500 1000 2500 --max 150 --seed 42 --iters 100

### Supported Parameters

| Flag        | Description                                                    | Default |
|-------------|----------------------------------------------------------------|---------|
| `--size`    | Matrix sizes to test (you can specify multiple)               | `2500`  |
| `--max`     | Desired maximum eigenvalue for the test matrices              | `100`   |
| `--seed`    | Seed for random matrix generation                             | `123`   |
| `--iters`    | Maximum iterations per algorithm                              | `300`   |

## 📊 Program Output Example

Each line reports the result of one algorithm on a given matrix:
    Gradient Descent: Time = 7.5555s | Iter = 300
    BFGS:             Time = 10.2529s | Iter = 129
    L-BFGS:           Time = 2.7036s  | Iter = 58


## Output Plots

Plots are saved in the current directory.
