import autograd.numpy as np
from ..optimization.single_experiment.select_optimizer import import_optimizer
from ..estimation.problem_generator import OptimizationProblemGenerator
from ..models.utils import dict_to_nparray, nparray_to_dict
from ..logging_ import logger


def get_bifurcation_point(
    branch: np.ndarray, model: object, optimizer_name: str = "scipy"
) -> np.ndarray:
    """
    Gets the exact bifurcation point from an approximation.

    Args:
        branch (np.ndarray): Bifurcation point approximation from the branches.
        model (object): Model details.
        optimizer_name (str): Name of the local optimizer.

    Returns:
        np.ndarray: Exact bifurcation point.

    Raises:
        ValueError: If the bifurcation type is unrecognized.
        RuntimeError: If the bifurcation point cannot be found.
    """
    parameter = model.controls["homotopy"]

    assert len(branch) == len(model.compartments) + 1
    c, p, h = nparray_to_dict(x=branch, model=model)

    jacobian_ = model.jacobian_(branch)
    eig_vals, eig_vecs = np.linalg.eig(jacobian_)
    mask = (eig_vals.real == min(eig_vals.real, key=abs)) & (eig_vals.imag >= 0)

    if model.bifurcation_type == "hopf":
        h["mu"] = eig_vals.imag[mask]
        h["v"] = eig_vecs.real[:, mask].squeeze()
        h["w"] = eig_vecs.imag[:, mask].squeeze()

    elif model.bifurcation_type == "saddle-node":
        h["h"] = eig_vecs.real[:, mask].squeeze()

    else:
        raise ValueError("Unrecognized bifurcation type!")

    model.mask["auxiliary_variables"] = True
    branch = dict_to_nparray(c=c, p=p, h=h, model=model)

    Optimizer = import_optimizer(optimizer_name)

    Objective = OptimizationProblemGenerator(model, include_singularity=True)
    objective_function = Objective.stack_functions

    Constraints = OptimizationProblemGenerator(
        model, include_steady_state=True, include_normalization=True
    )
    equality_constraints = Constraints.stack_functions

    logger.debug(f"Get the bifurcation point near {p[parameter]} using {optimizer_name} optimizer.")
    optimizer = Optimizer(
        objective=objective_function,
        x0=branch,
        constraints={"type": "eq", "fun": equality_constraints},
    )

    optimizer.minimize(method="SLSQP")

    if optimizer.result.success:
        solution = optimizer.result.x
        max_obj = np.linalg.norm(objective_function(solution), ord=np.inf)
        if not np.isclose(max_obj, 0):
            logger.warning(f"Objective function is satisfied only upto {max_obj:.3e}")
        _, p, _ = nparray_to_dict(x=solution, model=model)
        logger.info(f"Found a bifurcation point at {p[parameter]}.")
        return solution
    else:
        model.mask["auxiliary_variables"] = False
        raise RuntimeError("Could not find a bifurcation point!")
