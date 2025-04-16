def import_optimizer(optimizer: str):
    """
    Returns a single experiment optimizer.

    Args:
        optimizer (str): Name of the optimization method.

    Returns:
        object: Optimizer object.

    Raises:
        ValueError: If the optimizer name is unknown.
    """
    if optimizer == "osqp":
        from ...optimization.multi_experiment.osqp_optimizer import MultiExperimentOSQP

        return MultiExperimentOSQP

    elif optimizer == "gauss-newton":
        from ...optimization.multi_experiment.gauss_newton_optimizer import (
            MultiExperimentGaussNewton,
        )

        return MultiExperimentGaussNewton

    else:
        raise ValueError(f"Unknown optimizer name: {optimizer}")
