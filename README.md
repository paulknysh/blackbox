# blackbox: A Python module for parallel optimization of expensive black-box functions

## What is this?

Let's say you are an engineer/scientist who works with numerical simulations. Oftentimes you need to find optimal parameters of a given design/model. If you can construct a simple Python function, that takes a set of trial parameters, performs simulation, and outputs some scalar measure (of how good your design/model is), then the problem becomes a mathematical optimization, that can and should be automated. However, a corresponding function has no analytical expression (black-box) and is usually computationally expensive, which makes it challenging to deal with for common methods.

**blackbox** is a minimalistic and easy-to-use Python module that efficiently searches for a global optimum (minimum) of an expensive black-box function. It works based on a given (often limited) number of function calls and makes an efficient use of them (whether it's 10 function calls or 100) to find the best solution it can. It scales on multicore CPUs by performing several function evaluations in parallel, which results in a speedup equal to a number of cores available.

Code is multidimensional and currently handles only box-constrained search regions - each variable has its own independent range. A mathematical method behind the code is described in this [**arXiv note**](http://arxiv.org/pdf/1605.00998.pdf).

## How do I represent my objective function?

It simply needs to be wrapped into a Python function. An external numerical package can be accessed using system call:
```python
def fun(par):

  # modifying text file that contains design/model parameters
  ...
  # performing simulation in external package
  os.system(...)
  # reading results, calculating output
  ...
  
  return output
```
Here `par` is a vector of parameters (a Python list is OK) and `output` is a **non-negative** scalar measure of interest.

## How do I run the procedure?

Just like that (**minimizing** a given function):
```python
from blackbox import *


def fun(par):
  ...
  return output


if __name__ == '__main__':

  search(
  
    f=fun, # given function
	
    resfile='output.csv', # .csv file where iterations will be saved

    box=np.array([[-1.,1.],[-1.,1.]]), # range of values for each variable

    cores=4, # number of cores to be used

    n=8, # number of function calls on initial stage (global search)
    it=8 # number of function calls on subsequent stage (local search)
    
    )
```

## How about results?

Iterations are saved in .csv file with the following structure:

1st variable | 2nd variable | ... | n-th variable | Function value
--- | --- | --- | --- | ---
0.172 | 0.467 | ... | 0.205 | 0.107
0.164 | 0.475 | ... | 0.216 | 0.042
... | ... | ... | ... | ...

**Important**: For a number of reasons all variables (as well as a function value) are normalized into range [0,1]. If a given variable v has range [a,b], then 0 corresponds to a and 1 corresponds to b. Simple linear rescaling can be applied to obtain an absolute value needed.

Data from .csv file can be postprocessed and visualized using provided Mathematica script `output.m`.

## Author

Paul Knysh (paul.knysh@gmail.com)
