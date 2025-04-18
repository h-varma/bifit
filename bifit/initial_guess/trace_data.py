import autograd.numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from ..continuation.select_continuer import import_continuer
from ..estimation.problem_generator import OptimizationProblemGenerator
from ..models.utils import dict_to_nparray, nparray_to_dict
from ..postprocessing.plot_decorator import handle_plots


@handle_plots(plot_name="bifurcation_curve")
def trace_measured_bifurcations(
    x0: np.ndarray,
    model: object,
    continuer_name: str = "pseudo-arclength",
) -> tuple[np.ndarray, plt.Figure]:
    """
    Continues the bifurcation point to draw a two-parameter bifurcation diagram.

    Args:
        x0 (np.ndarray): Starting bifurcation point.
        model (object): Model details.
        continuer_name (str): Continuation method.

    Returns:
        tuple[np.ndarray, plt.Figure]: Set of bifurcation points in two parameters and the figure object.
    """
    Continuer = import_continuer(continuer_name)
    homotopy_parameter = model.controls["homotopy"]
    free_parameter = model.controls["free"]

    data = np.unique([d[homotopy_parameter] for d in model.data])

    c, p, h = nparray_to_dict(x=x0, model=model)
    p0 = p[homotopy_parameter]
    p_idx = len(model.compartments)

    model.parameters[free_parameter]["vary"] = True

    ContinuationProblem = OptimizationProblemGenerator(
        model=model,
        include_steady_state=True,
        include_singularity=True,
        include_normalization=True,
    )
    objective_function = ContinuationProblem.stack_functions

    x0 = dict_to_nparray(c=c, p=p, h=h, model=model)
    x0 = np.delete(x0, obj=p_idx)

    if continuer_name == "deflated":
        kwargs = {
            "p_min": np.min([0, 0.5 * min(data), p0, model.continuation_settings["h_min"]]),
            "p_max": np.max([2 * max(data), p0, model.continuation_settings["h_max"]]),
            "p_step": np.min(
                [0.5 * min(np.abs(np.diff(data))), model.continuation_settings["h_step"]]
            ),
            "max_failed_attempts": 1,
            "unique_indices": np.arange(0, len(model.compartments)),
        }
    elif continuer_name == "pseudo-arclength":
        kwargs = {
            "p_step": model.continuation_settings["h_step"],
            "max_iters": 1000,
            "newton_fun_tol": np.inf,
        }

    continuer = Continuer(func=objective_function, x0=x0, p0=p0, p_idx=p_idx, data=data, **kwargs)

    solutions_list = []
    fig, ax = plt.subplots()

    for parameter_value, solutions in zip(continuer.parameters, continuer.solutions):
        if continuer_name == "pseudo-arclength":
            solution = attach_parameter(p=parameter_value, s=solutions, model=model, ax=ax)
            solutions_list.append(solution)
        elif continuer_name == "deflated":
            solutions_list.append([])
            for solution in solutions:
                if not np.isnan(solution).all():
                    solution = attach_parameter(p=parameter_value, s=solution, model=model, ax=ax)
                    solutions_list[-1].append(solution)
                else:
                    solutions_list[-1].append(np.nan * np.ones(len(solution) + 1))
    ax.set_xlabel(homotopy_parameter, fontsize=15)
    ax.set_ylabel(free_parameter, fontsize=15)

    return solutions_list, fig


def attach_parameter(p: float, s: np.ndarray, model: object, ax: plt.Axes) -> np.ndarray:
    """
    Adds the homotopy parameter to the solution vector and plots it.

    Args:
        p (float): Homotopy parameter value.
        s (np.ndarray): Solution vector.
        model (object): Model details.
        ax (plt.Axes): Axis object.

    Returns:
        np.ndarray: Complete solution vector.
    """
    homotopy_parameter = model.controls["homotopy"]
    free_parameter = model.controls["free"]

    model.parameters[homotopy_parameter]["vary"] = False
    c_, p_, h_ = nparray_to_dict(x=s, model=model)
    p_[homotopy_parameter] = p
    ax.plot(p_[homotopy_parameter], p_[free_parameter], "ok", markersize=1)

    model.parameters[homotopy_parameter]["vary"] = True
    solution = dict_to_nparray(c=c_, p=p_, h=h_, model=model)
    return solution
