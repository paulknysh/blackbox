# blackbox: A Python module for parallel optimization of expensive black-box functions

## What is this?

Let's say you need to find optimal parameters of some computationally intensive system (for example, time-consuming simulation). If you can construct a simple Python function, that takes a set of trial parameters, performs evaluation, and returns some scalar measure of how good chosen parameters are, then the problem becomes a mathematical optimization. However, a corresponding function is expensive (one evaluation can take hours) and is a black-box (has input-output nature).

**blackbox** is a minimalistic and easy-to-use Python module that efficiently searches for a global optimum (minimum) of an expensive black-box function. User needs to provide a function, a search region (ranges of values for each input parameter) and a number of function evaluations available. A code scales well on clusters and multicore CPUs by performing all expensive function evaluations in parallel.

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
`par` is a vector of input parameters (a Python list), `output` is a scalar measure to be minimized.

## How do I run the procedure?

No installation is needed. Just place `blackbox.py` into your working directory. Main file should look like that:
```python
import blackbox as bb


def fun(par):
    return par[0]**2 + par[1]**2 # dummy 2D example


def main():
    bb.search(f=fun,  # given function
              box=[[-10., 10.], [-10., 10.]],  # range of values for each parameter (2D case)
              n=20,  # number of function calls on initial stage (global search)
              m=20,  # number of function calls on subsequent stage (local search)
              batch=4,  # number of calls that will be evaluated in parallel
              resfile='output.csv')  # text file where results will be saved


if __name__ == '__main__':
    main()
```
**Important:**
* All function calls are divided into batches that are evaluated in parallel. Total number of these parallel cycles is `(n+m)/batch`.
* `n` must be greater than the number of parameters, `m` must be greater than 1, `batch` should not exceed the number of CPU cores available.
* An optional parameter `executor=...` should be specified when calling `bb.search()` in case when code is used on a cluster with some custom parallel engine (ipyparallel, dask.distributed, pathos etc). `executor` should be an object that has a `map` method.

## How about results?

Iterations are sorted by function value (best solution is in the top) and saved in a text file with the following structure:

Parameter #1 | Parameter #2 | ... | Parameter #n | Function value
--- | --- | --- | --- | ---
+1.6355e+01 | -4.7364e+03 | ... | +6.4012e+00 | +1.1937e-04
... | ... | ... | ... | ...

## Author

Paul Knysh (paul.knysh@gmail.com)

I receive tons of useful feedback that helps me to improve the code. Feel free to email me if you have any questions or comments.

<p align="center">
  <img src="http://i.imgur.com/De7yibS.png">
</p>
