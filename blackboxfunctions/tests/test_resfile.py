import os
from .utils import f, run_search_min

def test_no_resfile():
    run_search_min(f, [[-10,10]], 10, 5, None)
    # Passes without crash
    assert True

def test_resfile_exists():
    filename = 'output.csv'
    run_search_min(f, [[-10,10]], 10, 5, filename)
    assert os.path.exists(filename)
    os.remove(filename)