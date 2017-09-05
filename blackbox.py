import sys
import multiprocessing as mp
import numpy as np
import scipy.optimize as op


def get_default_executor():
    """
    Provide a default executor (a context manager
    returning an object with a map method).

    This is the multiprocessing Pool object () for python3.

    The multiprocessing Pool in python2 does not have an __enter__
    and __exit__ method, this function provides a backport of the python3 Pool
    context manager.

    Returns
    -------
    Pool : executor-like object
        An object with context manager (__enter__, __exit__) and map method.
    """
    if (sys.version_info > (3, 0)):
        Pool = mp.Pool
        return Pool
    else:
        from contextlib import contextmanager
        from functools import wraps

        @wraps(mp.Pool)
        @contextmanager
        def Pool(*args, **kwargs):
            pool = mp.Pool(*args, **kwargs)
            yield pool
            pool.terminate()
        return Pool


def search(f, box, n, m, batch, resfile,
           rho0=0.5, p=1.0, nrand=10000, nrand_frac=0.05,
           executor=get_default_executor()):
    """
    Minimize given expensive black-box function and save results into text file.

    Parameters
    ----------
    f : callable
        The objective function to be minimized.
    box : list of lists
        List of ranges for each parameter.
    n : int
        Number of initial function calls.
    m : int
        Number of subsequent function calls.
    batch : int
        Number of function calls evaluated simultaneously (in parallel).
    resfile : str
        Text file to save results.
    rho0 : float, optional
        Initial "balls density".
    p : float, optional
        Rate of "balls density" decay (p=1 - linear, p>1 - faster, 0<p<1 - slower).
    nrand : int, optional
        Number of random samples that is generated for space rescaling.
    nrand_frac : float, optional
        Fraction of nrand that is actually used for space rescaling.
    executor : callable, optional
        Should have a map method and behave as a context manager.
        Allows the user to use various parallelisation tools
        as dask.distributed or pathos.
    """
    # space size
    d = len(box)

    # adjusting the number of function calls to the batch size
    if n % batch != 0:
        n = n - n % batch + batch

    if m % batch != 0:
        m = m - m % batch + batch

    # go from normalized values (unit cube) to absolute values (box)
    def cubetobox(x):
        return [box[i][0]+(box[i][1]-box[i][0])*x[i] for i in range(d)]

    # generating latin hypercube
    points = np.zeros((n, d+1))
    points[:, 0:-1] = latin(n, d)

    # initial sampling
    for i in range(n//batch):
        with executor() as e:
            points[batch*i:batch*(i+1), -1] = list(e.map(f, list(map(cubetobox, points[batch*i:batch*(i+1), 0:-1]))))

    # normalizing function values
    fmax = max(abs(points[:, -1]))
    points[:, -1] = points[:, -1]/fmax

    # volume of d-dimensional ball (r = 1)
    if d % 2 == 0:
        v1 = np.pi**(d/2)/np.math.factorial(d/2)
    else:
        v1 = 2*(4*np.pi)**((d-1)/2)*np.math.factorial((d-1)/2)/np.math.factorial(d)

    # subsequent iterations (current subsequent iteration = i*batch+j)
    T = np.identity(d)

    for i in range(m//batch):

        # refining scaling matrix T
        if d > 1:
            fit_noscale = rbf(points, np.identity(d))
            population = np.zeros((nrand, d+1))
            population[:, 0:-1] = np.random.rand(nrand, d)
            population[:, -1] = list(map(fit_noscale, population[:, 0:-1]))

            cloud = population[population[:, -1].argsort()][0:int(nrand*nrand_frac), 0:-1]
            eigval, eigvec = np.linalg.eig(np.cov(np.transpose(cloud)))
            T = [eigvec[:, j]/np.sqrt(eigval[j]) for j in range(d)]
            T = T/np.linalg.norm(T)

        # sampling next batch of points
        fit = rbf(points, T)
        points = np.append(points, np.zeros((batch, d+1)), axis=0)

        for j in range(batch):
            r = ((rho0*((m-1.-(i*batch+j))/(m-1.))**p)/(v1*(n+i*batch+j)))**(1./d)
            cons = [{'type': 'ineq', 'fun': lambda x, localk=k: np.linalg.norm(np.subtract(x, points[localk, 0:-1])) - r}
                    for k in range(n+i*batch+j)]
            while True:
                minfit = op.minimize(fit, np.random.rand(d), method='SLSQP', bounds=[[0., 1.]]*d, constraints=cons)
                if np.isnan(minfit.x)[0] == False:
                    break
            points[n+i*batch+j, 0:-1] = np.copy(minfit.x)

        with executor() as e:
            points[n+batch*i:n+batch*(i+1), -1] = list(e.map(f, list(map(cubetobox, points[n+batch*i:n+batch*(i+1), 0:-1]))))/fmax

    # saving results into text file
    points[:, 0:-1] = list(map(cubetobox, points[:, 0:-1]))
    points[:, -1] = points[:, -1]*fmax
    points = points[points[:, -1].argsort()]

    labels = [' par_'+str(i+1)+(7-len(str(i+1)))*' '+',' for i in range(d)]+[' f_value    ']
    np.savetxt(resfile, points, delimiter=',', fmt=' %+1.4e', header=''.join(labels), comments='')


def latin(n, d):
    """
    Build latin hypercube.

    Parameters
    ----------
    n : int
        Number of points.
    d : int
        Size of space.

    Returns
    -------
    lh : ndarray
        Array of points uniformly placed in d-dimensional unit cube.
    """
    # spread function
    def spread(points):
        return sum(1./np.linalg.norm(np.subtract(points[i], points[j])) for i in range(n) for j in range(n) if i > j)

    # starting with diagonal shape
    lh = [[i/(n-1.)]*d for i in range(n)]

    # minimizing spread function by shuffling
    minspread = spread(lh)

    for i in range(1000):
        point1 = np.random.randint(n)
        point2 = np.random.randint(n)
        dim = np.random.randint(d)

        newlh = np.copy(lh)
        newlh[point1, dim], newlh[point2, dim] = newlh[point2, dim], newlh[point1, dim]
        newspread = spread(newlh)

        if newspread < minspread:
            lh = np.copy(newlh)
            minspread = newspread

    return lh


def rbf(points, T):
    """
    Build RBF-fit for given points (see Holmstrom, 2008 for details) using scaling matrix.

    Parameters
    ----------
    points : ndarray
        Array of multi-d points with corresponding values [[x1, x2, .., xd, val], ...].
    T : ndarray
        Scaling matrix.

    Returns
    -------
    fit : callable
        Function that returns the value of the RBF-fit at a given point.
    """
    n = len(points)
    d = len(points[0])-1

    def phi(r):
        return r*r*r

    Phi = [[phi(np.linalg.norm(np.dot(T, np.subtract(points[i, 0:-1], points[j, 0:-1])))) for j in range(n)] for i in range(n)]

    P = np.ones((n, d+1))
    P[:, 0:-1] = points[:, 0:-1]

    F = points[:, -1]

    M = np.zeros((n+d+1, n+d+1))
    M[0:n, 0:n] = Phi
    M[0:n, n:n+d+1] = P
    M[n:n+d+1, 0:n] = np.transpose(P)

    v = np.zeros(n+d+1)
    v[0:n] = F

    sol = np.linalg.solve(M, v)
    lam, b, a = sol[0:n], sol[n:n+d], sol[n+d]

    def fit(x):
        return sum(lam[i]*phi(np.linalg.norm(np.dot(T, np.subtract(x, points[i, 0:-1])))) for i in range(n)) + np.dot(b, x) + a

    return fit
