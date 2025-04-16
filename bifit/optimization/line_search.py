from scipy.optimize import minimize_scalar
import autograd.numpy as np
from autograd import grad


def line_search(x: np.ndarray, dx: np.ndarray, func: callable, strategy: str) -> float:
    """
    Find the step length using the line search.

    Args:
        x (np.ndarray): Current point.
        dx (np.ndarray): Search direction.
        func (callable): Objective function.
        strategy (str): Line search strategy.

    Returns:
        float: Step length.

    Raises:
        ValueError: If the specified line search strategy is not supported.
        AssertionError: If the exact line search does not converge.
    """

    if strategy == "exact":

        def step_finder(t_):
            return func(x + t_ * dx)

        result = minimize_scalar(step_finder, method="Bounded", bounds=(0, 1))
        assert result.success, "Exact line search did not converge."
        return result.x

    elif strategy == "armijo-backtracking":
        f = func(x)
        df = grad(func)(x)
        t = 1
        beta = 0.5
        gamma = 1e-4
        while func(x + t * dx) > f + gamma * t * (df.T @ dx):
            t *= beta
        return t

    else:
        raise ValueError(f"{strategy} line search strategy is not supported.")
