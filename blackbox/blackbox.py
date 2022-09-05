import multiprocessing as mp
import numpy as np
import scipy.optimize as op
import logging
from typing import Callable


logging.basicConfig(
    format="%(levelname)-8s %(message)s %(asctime)s", datefmt="%m-%d %H:%M:%S"
)
logger = logging.getLogger("blackbox")
logger.setLevel(logging.INFO)


def minimize(
    f: Callable,
    domain: list,
    budget: int,
    batch: int,
    rho0: float = 0.5,
    p: float = 1.0,
    executor: Callable = mp.Pool,
) -> dict:
    """Minimize given expensive black-box function

    Args:
        f (Callable): objective function to be minimized
        domain (list): ranges for each parameter
        budget (int): total number of function calls available
        batch (int): number of function calls evaluated simultaneously (in parallel)
        rho0 (float, optional): initial "balls density". Defaults to 0.5
        p (float, optional): rate of "balls density" decay (p=1 - linear, p>1 - faster, 0<p<1 - slower). Defaults to 1.0
        executor (Callable, optional): should have a map method and behave as a context manager. Defaults to mp.Pool

    Returns:
        dict: a dictionary with results. Contains the following keys:
        "best_x": best solution
        "best_f": corresponding best function value
        "all_xs": all points that were evaluated
        "all_fs": all corresponding function values
    """
    # space size
    d = len(domain)

    # adjusting the budget to the batch size
    if budget % batch != 0:
        budget = budget - budget % batch + batch
        logger.info(f"budget was adjusted to be {budget}")

    # default global-vs-local assumption (50-50)
    n = budget // 2
    if n % batch != 0:
        n = n - n % batch + batch
    m = budget - n

    # n has to be greater than d
    if n <= d:
        logger.error("budget is not sufficient")
        return

    # go from normalized values (unit cube) to absolute values (box)
    def cubetobox(x):
        return [domain[i][0] + (domain[i][1] - domain[i][0]) * x[i] for i in range(d)]

    # generating R-sequence
    points = np.zeros((n, d + 1))
    points[:, 0:-1] = build_rseq(n, d)

    # initial sampling
    for i in range(n // batch):
        logger.info(
            f"evaluating batch {i+1}/{(n+m)//batch} (samples {i*batch+1}..{(i+1)*batch}/{n+m})"
        )

        with executor() as e:
            points[batch * i : batch * (i + 1), -1] = list(
                e.map(
                    f, list(map(cubetobox, points[batch * i : batch * (i + 1), 0:-1]))
                )
            )

    # normalizing function values
    fmax = max(abs(points[:, -1]))
    points[:, -1] = points[:, -1] / fmax

    v1 = compute_volume_unit_ball(d)

    # subsequent iterations (current subsequent iteration = i*batch+j)
    for i in range(m // batch):
        logger.info(
            f"evaluating batch {n//batch+i+1}/{(n+m)//batch} (samples {n+i*batch+1}..{n+(i+1)*batch}/{n+m})"
        )

        # sampling next batch of points
        fit = build_rbf(points)
        points = np.append(points, np.zeros((batch, d + 1)), axis=0)

        for j in range(batch):
            r = (
                (rho0 * ((m - 1.0 - (i * batch + j)) / (m - 1.0)) ** p)
                / (v1 * (n + i * batch + j))
            ) ** (1.0 / d)
            constraints = [
                {
                    "type": "ineq",
                    "fun": lambda x, localk=k: np.linalg.norm(
                        np.subtract(x, points[localk, 0:-1])
                    )
                    - r,
                }
                for k in range(n + i * batch + j)
            ]
            while True:
                minfit = op.minimize(
                    fit,
                    np.random.rand(d),
                    method="SLSQP",
                    bounds=[[0.0, 1.0]] * d,
                    constraints=constraints,
                )
                if np.isnan(minfit.x)[0] == False:
                    break
            points[n + i * batch + j, 0:-1] = np.copy(minfit.x)

        with executor() as e:
            points[n + batch * i : n + batch * (i + 1), -1] = (
                list(
                    e.map(
                        f,
                        list(
                            map(
                                cubetobox,
                                points[n + batch * i : n + batch * (i + 1), 0:-1],
                            )
                        ),
                    )
                )
                / fmax
            )

    # rescaling result
    points[:, 0:-1] = list(map(cubetobox, points[:, 0:-1]))
    points[:, -1] = points[:, -1] * fmax

    logger.info("DONE")

    return {
        "best_x": points[points[:, -1].argsort()][0, :-1],
        "best_f": points[points[:, -1].argsort()][0, -1],
        "all_xs": points[:, :-1],
        "all_fs": points[:, -1],
    }


def compute_volume_unit_ball(d: int) -> float:
    """computes volume of d-dimentional ball (r=1)

    Args:
        d (int): space dimension

    Returns:
        float: volume
    """
    if d % 2 == 0:
        v1 = np.pi ** (d / 2) / np.math.factorial(d / 2)
    else:
        v1 = (
            2
            * (4 * np.pi) ** ((d - 1) / 2)
            * np.math.factorial((d - 1) / 2)
            / np.math.factorial(d)
        )
    return v1


def build_rseq(n: int, d: int) -> np.ndarray:
    """build R-sequence (array of points uniformly placed in d-dimensional unit cube):
    http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/

    Args:
        n (int): number of points
        d (int): space dimention

    Returns:
        np.ndarray: R-sequence
    """
    phi = 2
    for i in range(10):
        phi = pow(1 + phi, 1.0 / (d + 1))

    alpha = np.array([pow(1.0 / phi, i + 1) for i in range(d)])
    rseq = np.array([(0.5 + alpha * (i + 1)) % 1 for i in range(n)])

    return rseq


def build_rbf(points: np.ndarray) -> Callable:
    """build RBF-fit for given points (see Holmstrom, 2008 for details)

    Args:
        points (np.ndarray): array of multi-d points with corresponding values
        [[x1, x2, .., xd, val], ...]

    Returns:
        Callable: constructed fit
    """
    n = len(points)
    d = len(points[0]) - 1

    def basis(r):
        return r * r * r

    phi = [
        [
            basis(np.linalg.norm(np.subtract(points[i, 0:-1], points[j, 0:-1])))
            for j in range(n)
        ]
        for i in range(n)
    ]

    p = np.ones((n, d + 1))
    p[:, 0:-1] = points[:, 0:-1]

    f = points[:, -1]

    m = np.zeros((n + d + 1, n + d + 1))
    m[0:n, 0:n] = phi
    m[0:n, n : n + d + 1] = p
    m[n : n + d + 1, 0:n] = np.transpose(p)

    v = np.zeros(n + d + 1)
    v[0:n] = f

    try:
        sol = np.linalg.solve(m, v)
    except:
        # helps with singular matrices
        logger.warning(
            "Singular matrix occurred during RBF-fit construction. RBF-fit might be inaccurate!"
        )
        sol = np.linalg.lstsq(m, v)[0]

    lam, b, a = sol[0:n], sol[n : n + d], sol[n + d]

    def fit(x):
        return (
            sum(
                lam[i] * basis(np.linalg.norm(np.subtract(x, points[i, 0:-1])))
                for i in range(n)
            )
            + np.dot(b, x)
            + a
        )

    return fit
