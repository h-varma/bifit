import autograd.numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from ..continuation.select_continuer import import_continuer
from ..models.utils import nparray_to_dict
from ..postprocessing.plot_decorator import handle_plots


@handle_plots(plot_name="steady_state_curve")
def compute_steady_state_curve(
    x0: np.ndarray,
    model: dataclass,
    continuer_name: str = "deflated",
) -> tuple[list[np.ndarray], plt.Figure]:
    """
    Draws a steady state curve starting from the steady state x0.

    Args:
        x0 (np.ndarray): Starting point.
        model (dataclass): Details of the model.
        continuer_name (str): Continuation method.

    Returns:
        tuple[list[np.ndarray], plt.Figure]: Results of the continuation and the figure object.

    Raises:
        ValueError: If the bifurcation type is unrecognized.
    """

    # continue the steady state solutions along the homotopy parameter
    parameter = model.controls["homotopy"]

    model.mask = {
        "compartments": True,
        "controls": True,
        "auxiliary_variables": False,
        "global_parameters": False,
    }
    model.parameters[parameter]["vary"] = True

    Continuer = import_continuer(continuer_name)

    steady_states = Continuer(
        func=model.rhs_,
        x0=x0,
        p0=model.parameters[parameter]["value"],
        p_min=model.continuation_settings["h_min"],
        p_max=model.continuation_settings["h_max"],
        p_step=model.continuation_settings["h_step"],
        p_idx=len(x0),
        unique_indices=np.arange(0, len(model.compartments)),
    )

    idx = model.compartments.index(model.to_plot)
    fig, ax = plt.subplots()
    for parameter_value, solutions in zip(steady_states.parameters, steady_states.solutions):
        for solution in solutions:
            ax.plot(parameter_value, solution[idx], "ok", markersize=1)
    ax.set_xlabel(model.controls["homotopy"], fontsize=15)
    ax.set_ylabel(model.to_plot, fontsize=15)

    # detect the bifurcation point from the continuation results
    if model.bifurcation_type == "hopf":
        branches = steady_states.detect_hopf_bifurcation(parameter=parameter)
    elif model.bifurcation_type == "saddle-node":
        branches = steady_states.detect_saddle_node_bifurcation(parameter=parameter)
    else:
        raise ValueError("Unrecognized bifurcation type!")

    # filter out branches that are multiples of states to be excluded
    if len(model.exclude_states) > 0:
        for i, branch in enumerate(branches):
            branch = np.fromiter(nparray_to_dict(branch, model=model)[0].values(), dtype=float)
            for state in model.exclude_states:
                lhs = np.dot(branch, state) ** 2
                rhs = np.dot(branch, branch) * np.dot(state, state)
                if np.isclose(lhs, rhs):
                    branches.pop(i)

    return branches, fig
