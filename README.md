# blackbox: A Python module for parallel optimization of expensive black-box functions

## What is this?

A minimalistic and easy-to-use Python module that efficiently searches for a global minimum of an expensive black-box function (e.g. optimal hyperparameters of simulation, neural network or anything that takes significant time to run). User needs to provide a function, a search domain (ranges of each input parameter) and a total number of function calls available. A code scales well on multicore CPUs and clusters: all function calls are divided into batches and each batch is evaluated in parallel.

A mathematical method behind the code is described in this arXiv note (there were few updates to the method recently): https://arxiv.org/pdf/1605.00998.pdf

Don't forget to cite this note if you are using method/code.

## Demo

<img src="https://i.imgur.com/kkagLKR.png">

(a) - demo function (unknown to a method).

(b) - running a procedure using 15 evaluations.

(c) - running a procedure using 30 evaluations.

## Installation

To install locally run either:

`uv sync --extra dev`

or

`pip install ".[dev]"`

## Testing, linting, and formatting

Run the following from the repository root:

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format .
```

CI also runs these steps automatically on every push and pull request (with `ruff format --check` instead of `ruff format`).

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


if __name__ == "__main__":
    result = bb.minimize(
        f=fun,  # given function
        domain=[[-5, 5], [-5, 5]],  # ranges of each parameter
        budget=20,  # total number of function calls available
        batch=4,  # number of calls that will be evaluated in parallel
    )
    # best result (x and function value)
    print(result["best_x"])
    print(result["best_f"])

    # the entire history of evaluations
    # print(result["all_xs"])
    # print(result["all_fs"])
```
**Important:**
* All function calls are divided into batches and each batch is evaluated in parallel. Total number of batches is `ceil(budget/batch)` (the budget is automatically rounded up to a multiple of `batch`). The value of `batch` should correspond to the number of available computational units.
* An optional parameter `executor = ...` should be specified within `bb.minimize()` in case when custom parallel engine is used (ipyparallel, dask.distributed, pathos etc). `executor` must be a **callable** (e.g. a class or factory) that, when called with no arguments, returns a context-managing object exposing a `map` method (it is invoked as `with executor() as e:`). The default is `multiprocessing.Pool`.
* The default `executor` is `multiprocessing.Pool`, which pickles the objective function `f` to send it to workers. This means `f` must be picklable: a `lambda` or a closure defined in a REPL/notebook will raise a `PicklingError`. Use a top-level function (as in the example above) or pass a custom serial executor / `dask`/`ipyparallel` executor that avoids pickling.

## Results

`bb.minimize()` returns a dictionary with the following keys:
- `"best_x"` - best iteration
- `"best_f"` - corresponding function value
- `"all_xs"` - all iterations
- `"all_fs"` - corresponding function values

## License

MIT