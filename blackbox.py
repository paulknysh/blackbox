import numpy as np
import multiprocessing as mp
from scipy.optimize import minimize


def search(f, box, n, it, cores, resfile,
           rho0=0.5, p=1.0, nrand=10000, vf=0.05):
    """
    Minimize given expensive black-box function and write results to text file.

    Parameters
    ----------
    f : callable
        The objective function to be minimized.
    box : list of lists
        List of ranges for each parameter.
    n : int
        Number of initial function calls.
    it : int
        Number of subsequent function calls.
    cores : int
        Number of cores available.
    resfile : str
        Name of text file to save iterations.
    rho0 : float, optional
        Initial "balls density".
    p : float, optional
        Rate of "balls density" decay (p=1 - linear, p>1 - faster, 0<p<1 - slower).
    nrand : int, optional
        Number of random samples that is used for space rescaling.
    vf : float, optional
        Fraction of nrand that is actually used for space rescaling.
    """
    # space size
    d = len(box)

    # adjust the number of iterations to the number of cores
    if n % cores:
        n = n - n % cores + cores

    if it % cores:
        it = it - it % cores + cores

    # scales a given point from a unit cube to box
    def cubetobox(pt):
        res = np.zeros(d)
        for i in range(d):
            res[i] = box[i][0] + (box[i][1] - box[i][0]) * pt[i]
        return res

    # generating latin hypercube
    pts = np.zeros((n, d+1))
    lh = latin(n, d)

    for i in range(n):
        for j in range(d):
            pts[i, j] = lh[i, j]

    # initial sampling
    for i in range(n//cores):
        pts[cores*i:cores*(i+1), -1] = pmap(f, list(map(cubetobox, pts[cores*i:cores*(i+1), 0: -1])), cores)

    # rescaling function values
    fmax = max(abs(pts[:, -1]))
    pts[:, -1] = pts[:, -1]/fmax

    # volume of d-dimensional ball (r = 1)
    if not d % 2:
        v1 = np.pi**(d/2)/np.math.factorial(d/2)
    else:
        v1 = 2*(4*np.pi)**((d-1)/2)*np.math.factorial((d-1)/2)/np.math.factorial(d)

    # iterations (current iteration m is equal to h*cores+i)
    T = np.identity(d)

    for h in range(it//cores):
        # refining scaling matrix T
        if d > 1:
            pcafit = rbf(pts, np.identity(d))
            cover = np.zeros((nrand, d+1))
            cover[:, 0:-1] = np.random.rand(nrand, d)
            for i in range(nrand):
                cover[i, -1] = pcafit(cover[i, 0: -1])

            cloud = cover[cover[:, -1].argsort()][0:int(np.ceil(vf*nrand)), 0:-1]
            eigval, eigvec = np.linalg.eig(np.cov(np.transpose(cloud)))

            T = np.zeros((d, d))
            for i in range(d):
                T[i] = eigvec[:, i]/np.sqrt(eigval[i])
            T = T/np.linalg.norm(T)

        # sampling next batch of points
        fit = rbf(pts, T)
        pts = np.append(pts, np.zeros((cores, d+1)), axis=0)

        for i in range(cores):
            r = ((rho0*((it-1.-(h*cores+i))/(it-1.))**p)/(v1*(n+(h*cores+i))))**(1./d)
            cons = [{'type': 'ineq', 'fun': lambda x, localj=j: np.linalg.norm(np.subtract(x, pts[localj, 0:-1])) - r}
                    for j in range(n+h*cores+i)]
            while True:
                res = minimize(fit, np.random.rand(d), method='SLSQP', bounds=[[0., 1.]]*d, constraints=cons)
                if np.isnan(res.x)[0] == False:
                    break
            pts[n+h*cores+i, 0:-1] = np.copy(res.x)

        pts[n+cores*h:n+cores*(h+1), -1] = pmap(f, list(map(cubetobox, pts[n+cores*h:n+cores*(h+1), 0:-1])), cores)/fmax

    # saving results into external file
    pts[:, 0:-1] = list(map(cubetobox, pts[:, 0:-1]))
    pts[:, -1] = pts[:, -1]*fmax
    pts = pts[pts[:, -1].argsort()]

    labels = [' par_'+str(i+1)+',' for i in range(d)]+[' fun_val']
    labels = [labels[i]+(13-len(labels[i]))*' ' for i in range(d+1)]

    np.savetxt(resfile, pts, delimiter=',', fmt=' %+1.4e', header=''.join(labels), comments='')


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
    pts : ndarray
        Array of points uniformly placed in d-dimensional unit cube.
    """
    # starting with diagonal shape
    pts = np.ones((n, d))

    for i in range(n):
        pts[i] = pts[i]*i/(n-1.)

    # spread function
    def spread(p):
        s = 0.
        for i in range(n):
            for j in range(n):
                if i > j:
                    s = s+1./np.linalg.norm(np.subtract(p[i], p[j]))
        return s

    # minimizing spread function by shuffling
    currminspread = spread(pts)

    for m in range(1000):

        p1 = np.random.randint(n)
        p2 = np.random.randint(n)
        k = np.random.randint(d)

        newpts = np.copy(pts)
        newpts[p1, k], newpts[p2, k] = newpts[p2, k], newpts[p1, k]
        newspread = spread(newpts)

        if newspread < currminspread:
            pts = np.copy(newpts)
            currminspread = newspread

    return pts


def rbf(pts, T):
    """
    Build RBF-fit for given points (see Holmstrom, 2008 for details) using
    scaling matrix.

    Parameters
    ----------
    pts : ndarray
        Array of multi-d points with corresponding values [[x1, x2, .., xd, val], ...].
    T : ndarray
        Scaling matrix.

    Returns
    -------
    fit : callable
        Function that returns the value of the RBF-fit at a given point.
    """
    n = len(pts)
    d = len(pts[0])-1

    def phi(r):
        return r*r*r

    Phi = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Phi[i, j] = phi(np.linalg.norm(np.dot(T, np.subtract(pts[i, 0:-1], pts[j, 0:-1]))))

    P = np.ones((n, d+1))
    for i in range(n):
        P[i, 0:-1] = pts[i, 0:-1]

    F = np.zeros(n)
    for i in range(n):
        F[i] = pts[i, -1]

    M = np.zeros((n+d+1, n+d+1))
    M[0:n, 0:n] = Phi
    M[0:n, n:n+d+1] = P
    M[n:n+d+1, 0:n] = np.transpose(P)

    v = np.zeros(n+d+1)
    v[0:n] = F

    sol = np.linalg.solve(M, v)

    lam = sol[0:n]
    b = sol[n:n+d]
    a = sol[n+d]

    def fit(z):
        res = 0.
        for i in range(n):
            res = res+lam[i]*phi(np.linalg.norm(np.dot(T, np.subtract(z, pts[i, 0:-1]))))
        res = res+np.dot(b, z)+a
        return res

    return fit


def pmap(f, batch, n):
    """
    Map a function on a batch of arguments in a parallel way using multiple cores.

    Parameters
    ----------
    f : callable
       Function.
    batch : list
       List of arguments.
    n : int
       Number of cores.

    Returns
    -------
    res : list
        List of corresponding values.
    """
    pool = mp.Pool(processes=n)
    res = pool.map(f, batch)
    pool.close()
    pool.join()
    return res
