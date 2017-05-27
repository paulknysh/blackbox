# blackbox: A Python module for parallel optimization of expensive black-box functions

## What is this?

Let's say you need to find optimal parameters of some computationally intensive system (for example, time-consuming simulation). If you can construct a simple Python function, that takes a set of trial parameters, performs evaluation, and returns some scalar measure of how good chosen parameters are, then the problem becomes a mathematical optimization. However, a corresponding function is expensive (one evaluation can take hours) and is a black-box (has input-output nature).

**blackbox** is a minimalistic and easy-to-use Python module that efficiently searches for a global optimum (minimum) of an expensive black-box function. User needs to provide a function, a search region (ranges of values for each input parameter) and a number of function evaluations available. A code is designed to work on multicore CPUs by performing several function evaluations in parallel, which results in a speedup equal to a number of cores.

A mathematical method behind the code is described in this arXiv note: https://arxiv.org/pdf/1605.00998.pdf

Feel free to cite this note if you are using method/code in your research.

## How do I represent my objective function?

It simply needs to be wrapped into a Python function. If an external application is used, it can be accessed using system call:
```python
def fun(par):

    # running external application for given set of parameters
    os.system(...)
    
    # calculating output
    ...
    
    return output
```
`par` is a vector of input parameters (a Python list), `output` is a **non-negative** scalar measure to be **minimized**.

## How do I run the procedure?

No installation is needed. Just place `blackbox.py` into your working directory. Main file should look like that:
```python
import blackbox as bb


def fun(par):
    ...
    return output


def main():
    bb.search(f=fun,  # given function
              box=[[-10., 10.], [-10., 10.]],  # range of values for each parameter (2D example)
              n=8,  # number of function calls on initial stage (global search)
              it=8,  # number of function calls on subsequent stage (local search)
              cores=4,  # number of cores to be used
              resfile='output.csv')  # .csv file where iterations will be saved


if __name__ == '__main__':
    main()
```
**Important:** `n` must be greater than the number of parameters, `it` must be greater than 1. Both `n` and `it` are expected to be divisible by `cores` (if not, code will adjust them automatically).

## How about results?

Iterations are saved in .csv file with the following structure:

Parameter #1 | Parameter #2 | ... | Parameter #n | Function value
--- | --- | --- | --- | ---
0.139 | 0.488 | ... | 0.205 | 0.637
... | ... | ... | ... | ...
0.042 | 0.042 | ... | 0.042 | 0.001

**Important:** In output file all parameters and function value are normalized into range [0, 1]: 0 corresponds to minimum value, 1 corresponds to maximum value. This should help user to analyze/interpret evolution of parameters and function values more objectively, in relative sense. Once desired set of parameters is chosen, a simple linear rescaling should be applied to obtain absolute values.

## Author

Paul Knysh (paul.knysh@gmail.com)

I receive tons of useful feedback that helps me to improve the code. Feel free to email me if you have any questions or comments.

<p align="center">
  <img src="http://i.imgur.com/De7yibS.png">
</p>
