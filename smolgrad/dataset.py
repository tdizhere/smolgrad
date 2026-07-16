from engine import Value
from sklearn.datasets import (
    make_moons,
    make_circles,
    make_blobs,
    make_regression,
    make_classification,
)
import random
import math


def _wrap(X, y, as_value=True):
    """
    Converts NumPy arrays into Value objects.

    Parameters
    ----------
    X : array-like
    y : array-like
    as_value : bool

    Returns
    -------
    (xs, ys)
    """

    if not as_value:
        return X, y

    xs = [[Value(float(v)) for v in row] for row in X]
    ys = [Value(float(v)) for v in y]

    return xs, ys


def xor(as_value=True):
    X = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]

    y = [0.0, 1.0, 1.0, 0.0]

    return _wrap(X, y, as_value)


def linear(
    samples=100,
    slope=2.0,
    intercept=1.0,
    noise=0.0,
    as_value=True,
):
    X = []
    y = []

    for _ in range(samples):
        x = random.uniform(-1, 1)
        target = slope * x + intercept + random.gauss(0, noise)

        X.append([x])
        y.append(target)

    return _wrap(X, y, as_value)

#currently not working and giving computation graph recursion error
def quadratic(
    samples=100,
    noise=0.0,
    as_value=True,
):
    X = []
    y = []

    for _ in range(samples):
        x = random.uniform(-2, 2)
        target = x**2 + random.gauss(0, noise)

        X.append([x])
        y.append(target)

    return _wrap(X, y, as_value)


def sine(
    samples=100,
    noise=0.0,
    as_value=True,
):
    X = []
    y = []

    for _ in range(samples):
        x = random.uniform(-math.pi, math.pi)
        target = math.sin(x) + random.gauss(0, noise)

        X.append([x])
        y.append(target)

    return _wrap(X, y, as_value)


def moons(
    samples=100,
    noise=0.1,
    random_state=42,
    as_value=True,
):
    X, y = make_moons(
        n_samples=samples,
        noise=noise,
        random_state=random_state,
    )

    return _wrap(X, y, as_value)


def circles(
    samples=100,
    noise=0.1,
    factor=0.5,
    random_state=42,
    as_value=True,
):
    X, y = make_circles(
        n_samples=samples,
        noise=noise,
        factor=factor,
        random_state=random_state,
    )

    return _wrap(X, y, as_value)


def blobs(
    samples=100,
    centers=3,
    cluster_std=1.0,
    random_state=42,
    as_value=True,
):
    result = make_blobs(
        n_samples=samples,
        cluster_std=cluster_std,
        random_state=random_state,
        centers=centers,
        return_centers=False,
        )
    X, y = result[:2]
    return _wrap(X, y, as_value)


def regression(
    samples=100,
    features=1,
    noise=0.0,
    random_state=42,
    as_value=True,
):
    result  = make_regression(
        n_samples=samples,
        n_features=features,
        noise=noise,
        random_state=random_state,
        coef=False,
    )

    X, y = result[:2]
    return _wrap(X, y, as_value)


def classification(
    samples=100,
    features=2,
    classes=2,
    random_state=42,
    as_value=True,
):
    X, y = make_classification(
        n_samples=samples,
        n_features=features,
        n_informative=features,
        n_redundant=0,
        n_classes=classes,
        random_state=random_state,
    )

    return _wrap(X, y, as_value)