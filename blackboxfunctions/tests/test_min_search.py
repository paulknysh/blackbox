from .utils import f, run_search_min

def test_min_search():
    """
    Test that min_search returns good enough test params that
    minimize the function. For 20 function calls the evaluation of 
    f should be in [-2, 2]
    """
    error = 2.0
    best_params = run_search_min(f, [[-10,10]], 20, 5, None)
    # result should be within error
    assert -error <= f(best_params) <= error