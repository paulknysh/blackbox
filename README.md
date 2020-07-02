# blackbox: A Python module for parallel optimization of expensive black-box functions

## What is this?

A minimalistic and easy-to-use Python module that efficiently searches for a global minimum of an expensive black-box function (e.g. optimal hyperparameters of simulation, neural network or anything that takes significant time to run). User needs to provide a function, a search domain (ranges of each input parameter) and a total number of function calls available. A code scales well on multicore CPUs and clusters: all function calls are divided into batches and each batch is evaluated in parallel.

A mathematical method behind the code is described in this arXiv note (there were few updates to the method recently): https://arxiv.org/pdf/1605.00998.pdf

Don't forget to cite this note if you are using method/code.

## Demo

<img src="http://i.imgur.com/kkagLKR.png">

(a) - demo function (unknown to a method).

(b) - running a procedure using 15 evaluations.

(c) - running a procedure using 30 evaluations.

## How do I represent my objective function?

It simply needs to be wrapped into a Python function. An external application, if any, can be accessed using system call.
```python
def fun(par):
    ...
    return output
```
`par` is a vector of input parameters (a Python list), `output` is a scalar value to be minimized.

## How do I run the procedure?

Just place `blackbox.py` into your working directory. Main file should look like this:
```python
import blackbox as bb


def fun(par):
    return par[0]**2 + par[1]**2  # dummy example


best_params = bb.search_min(f = fun,  # given function
                            domain = [  # ranges of each parameter
                                [-10., 10.],
                                [-10., 10.]
                                ],
                            budget = 40,  # total number of function calls available
                            batch = 4,  # number of calls that will be evaluated in parallel
                            resfile = 'output.csv')  # text file where results will be saved
```
**Important:**
* All function calls are divided into batches and each batch is evaluated in parallel. Total number of batches is `budget/batch`. The value of `batch` should correspond to the number of available computational units.
* An optional parameter `executor = ...` should be specified within `bb.search_min()` in case when custom parallel engine is used (ipyparallel, dask.distributed, pathos etc). `executor` should be an object that has a `map` method.

## How about results?

In addition to `search_min()` returning list of optimal parameters, all trials are sorted by function value (best ones at the top) and saved in a text file with the following structure:

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
