def import_optimizer(optimizer: str):
    """
    Return a single experiment optimizer.

    Args:
        optimizer (str): Name of the optimization method.

    Returns:
        object: Optimizer object.

    Raises:
        ValueError: If the optimizer name is unknown.
    """
    if optimizer == "scipy":
        from ...optimization.single_experiment.scipy_optimizer import ScipyOptimizer

        return ScipyOptimizer

    elif optimizer == "gauss-newton":
        from ...optimization.single_experiment.gauss_newton_optimizer import GaussNewtonOptimizer

        return GaussNewtonOptimizer

    else:
        raise ValueError(f"Unknown optimizer name: {optimizer}")
