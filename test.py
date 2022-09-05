import blackbox as bb


def fun(x):
    return (x[0] - 1) ** 2 + (x[1] - 1) ** 2


if __name__ == "__main__":
    domain = [[-5, 5], [-5, 5]]
    result = bb.minimize(f=fun, domain=domain, budget=20, batch=4)

    print(result["best_x"])
    print(result["best_f"])
