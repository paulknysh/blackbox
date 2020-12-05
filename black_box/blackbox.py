import sys
import multiprocessing as mp
import numpy as np
import scipy.optimize as op
import datetime


def search_min(f, domain, budget, batch, resfile,
               rho0=0.5, p=1.0,
               executor=mp.Pool):
    """
    Minimize given expensive black-box function and save results into text file.

    Parameters
    ----------
    f : callable
        The objective function to be minimized.
    domain : list of lists
        List of ranges for each parameter.
    budget : int
        Total number of function calls available.
    batch : int
        Number of function calls evaluated simultaneously (in parallel).
    resfile : str
        Text file to save results.
    rho0 : float, optional
        Initial "balls density".
    p : float, optional
        Rate of "balls density" decay (p=1 - linear, p>1 - faster, 0<p<1 - slower).
    executor : callable, optional
        Should have a map method and behave as a context manager.
        Allows the user to use various parallelisation tools
        as dask.distributed or pathos.

    Returns
    -------
    ndarray
        Optimal parameters.
    """
    # space size
    d = len(domain)

    # adjusting the budget to the batch size
    if budget % batch != 0:
        budget = budget - budget % batch + batch
        print('[blackbox] FYI: budget was adjusted to be ' + str(budget))

    # default global-vs-local assumption (50-50)
    n = budget//2
    if n % batch != 0:
        n = n - n % batch + batch
    m = budget-n

    # n has to be greater than d
    if n <= d:
        print('[blackbox] ERROR: budget is not sufficient')
        return

    # go from normalized values (unit cube) to absolute values (box)
    def cubetobox(x):
        return [domain[i][0]+(domain[i][1]-domain[i][0])*x[i] for i in range(d)]

    # generating R-sequence
    points = np.zeros((n, d+1))
    points[:, 0:-1] = rseq(n, d)

    # initial sampling
    for i in range(n//batch):
        print('[blackbox] evaluating batch %s/%s (samples %s..%s/%s) @ ' % (i+1, (n+m)//batch, i*batch+1, (i+1)*batch, n+m) + \
        str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' ...')

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

    for i in range(m//batch):
        print('[blackbox] evaluating batch %s/%s (samples %s..%s/%s) @ ' % (n//batch+i+1, (n+m)//batch, n+i*batch+1, n+(i+1)*batch, n+m) + \
        str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' ...')

        # sampling next batch of points
        fit = rbf(points)
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

    print('[blackbox] DONE: see results in ' + resfile + ' @ ' + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    return points[0, 0:-1]


def rseq(n, d):
    """
    Build R-sequence (http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/).

    Parameters
    ----------
    n : int
        Number of points.
    d : int
        Size of space.

    Returns
    -------
    points : ndarray
        Array of points uniformly placed in d-dimensional unit cube.
    """
    phi = 2
    for i in range(10):
        phi = pow(1+phi, 1./(d+1))

    alpha = np.array([pow(1./phi, i+1) for i in range(d)])

    points = np.array([(0.5 + alpha*(i+1)) % 1 for i in range(n)])

    return points


def rbf(points):
    """
    Build RBF-fit for given points (see Holmstrom, 2008 for details).

    Parameters
    ----------
    points : ndarray
        Array of multi-d points with corresponding values [[x1, x2, .., xd, val], ...].

    Returns
    -------
    fit : callable
        Function that returns the value of the RBF-fit at a given point.
    """
    n = len(points)
    d = len(points[0])-1

    def phi(r):
        return r*r*r

    Phi = [[phi(np.linalg.norm(np.subtract(points[i, 0:-1], points[j, 0:-1]))) for j in range(n)] for i in range(n)]

    P = np.ones((n, d+1))
    P[:, 0:-1] = points[:, 0:-1]

    F = points[:, -1]

    M = np.zeros((n+d+1, n+d+1))
    M[0:n, 0:n] = Phi
    M[0:n, n:n+d+1] = P
    M[n:n+d+1, 0:n] = np.transpose(P)

    v = np.zeros(n+d+1)
    v[0:n] = F

    try:
        sol = np.linalg.solve(M, v)
    except:
        # might help with singular matrices
        print('Singular matrix occurred during RBF-fit construction. RBF-fit might be inaccurate!')
        sol = np.linalg.lstsq(M, v)[0]

    lam, b, a = sol[0:n], sol[n:n+d], sol[n+d]

    def fit(x):
        return sum(lam[i]*phi(np.linalg.norm(np.subtract(x, points[i, 0:-1]))) for i in range(n)) + np.dot(b, x) + a

    return fit
