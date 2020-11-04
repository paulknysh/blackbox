import blackboxfunctions as bb

def f(par):
    return par[0]**2

def run_search_min(f, domain, budget, batch, resfile):
    return bb.search_min(
        f=f, domain=domain, budget =budget,
        batch=batch, resfile=resfile
    )

def get_bb_logger():
    return bb.blackbox._LOGGER