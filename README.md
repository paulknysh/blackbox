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

## Installation

Have `poetry` installed (https://python-poetry.org/docs/#installation). Then run:

`poetry install`

## Objective function

Simply needs to be wrapped into a Python function.
```python
def fun(par):
    ...
    return output
```
`par` is a vector of input parameters (a Python list), `output` is a scalar value to be minimized.

## Running the procedure

```python
import blackbox as bb


def fun(x):
    return (x[0] - 1) ** 2 + (x[1] - 1) ** 2


if __name__ == '__main__':
    result = bb.minimize(f=fun, # given function
        domain=[[-5, 5], [-5, 5]], # ranges of each parameter
        budget=20, # total number of function calls available
        batch=4 # number of calls that will be evaluated in parallel
    )
    # best result (x and function value)
    print(result["best_x"])
    print(result["best_f"])

    # the entire history of evaluations
    # print(result["all_xs"])
    # print(result["all_fs"])
```
**Important:**
* All function calls are divided into batches and each batch is evaluated in parallel. Total number of batches is `budget/batch`. The value of `batch` should correspond to the number of available computational units.
* An optional parameter `executor = ...` should be specified within `bb.minimize()` in case when custom parallel engine is used (ipyparallel, dask.distributed, pathos etc). `executor` should be an object that has a `map` method.

## Results

`bb.minimize()` returns a dictionary with the following keys:
- `"best_x"` - best iteration
- `"best_f"` - corresponding function value
- `"all_xs"` - all iterations
- `"all_fs"` - corresponding function values

## Author

Paul Knysh (paul.knysh at gmail dot com)

<p align="center">
  <img src="http://i.imgur.com/De7yibS.png">
</p>
