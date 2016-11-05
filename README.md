# blackbox: A Python module for parallel optimization of expensive black-box functions

## What is this?

Let's say you work with computationally intensive system and you want to find optimal parameters of that system. If you can construct a simple Python function, that takes a set of trial parameters, performs evaluation, and outputs some scalar measure of how good chosen parameters are (in some sense), then the problem becomes a mathematical optimization and can be automated. However, a corresponding function is expensive (one evaluation can take hours) and is a black-box (has input-output nature). Therefore, there is a need for a method that can optimize such functions using limited number of function evaluations.

**blackbox** is a minimalistic and easy-to-use Python module that efficiently searches for a global optimum (minimum) of an expensive black-box function. It works based on a given limited number of function evaluations and makes an efficient use of them to find the best solution it can. It scales on multicore CPUs by performing several function evaluations in parallel, which results in a speedup equal to a number of cores available.

A mathematical method behind the code is described in this [**arXiv note**](http://arxiv.org/pdf/1605.00998.pdf).

## How do I represent my objective function?

It simply needs to be wrapped into a Python function. An external application can be accessed using system call:
```python
def fun(par):

    # setting parameters
    ...
    # running external application
    os.system(...)
    # calculating output
    ...
    
    return output
```
Here `par` is a vector of parameters (a Python list is OK) and `output` is a **non-negative** scalar measure to be **minimized**.

## How do I run the procedure?

Just like that (minimizing a function):
```python
import blackbox as bb


def fun(par):
    ...
    return output


def main():
    bb.search(f=fun, # given function
              box=[[-10.,10.],[-10.,10.]], # range of values for each parameter
              n=8, # number of function calls on initial stage (global search)
              it=8, # number of function calls on subsequent stage (local search)
              cores=4, # number of cores to be used
              resfile='output.csv') # .csv file where iterations will be saved


if __name__ == '__main__':
    main()
```
`n` must be greater than number of parameters and `it` must be greater than 1. Both `n` and `it` are expected to be divisible by `cores` (if not, code will adjust them automatically).

## How about results?

Iterations are saved in .csv file with the following structure:

1st parameter | 2nd parameter | ... | n-th parameter | Function value
--- | --- | --- | --- | ---
0.172 | 0.467 | ... | 0.205 | 0.107
0.164 | 0.475 | ... | 0.216 | 0.042
... | ... | ... | ... | ...

**Important**: For a number of reasons all parameters (as well as a function value) are normalized into range [0,1]. If a given parameter has range [a,b], then 0 corresponds to a and 1 corresponds to b. Simple linear rescaling can be applied to obtain an absolute value needed.

Data from .csv file can be visualized using provided Mathematica script `output.m`.

## Author

Paul Knysh (paul.knysh@gmail.com)
