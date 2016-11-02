import blackbox as bb

def fun(par):
    x = par[0]
    y = par[1]
    return abs(x-y) + ((x+y-1)/3)**2

bb.search(f=fun,
          box=[[-1.,1.],[-1.,1.]],
          n=8,
          it=8,
          cores=1,
          resfile='output.csv')
