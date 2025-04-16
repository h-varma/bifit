import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root
from ..logging_ import logger
from ..postprocessing.plot_decorator import handle_plots


@handle_plots(plot_name="steady_state")
def find_steady_state(model: object) -> tuple[np.ndarray, plt.Figure]:
    """
    Integrates and solves the model equations to get a steady state solution.

    Args:
        model (object): Instance of the Model object.

    Returns:
        tuple[np.ndarray, plt.Figure]: Steady state solution and the figure object.

    Raises:
        RuntimeError: If the model equations cannot be solved to find the steady state.
    """

    # integrate the model equations to get a solution estimate
    y0 = model.initial_state
    t_span = model.integration_interval

    model.mask["compartments"] = True
    logger.debug("Integrate the model equations to get a solution estimate.")
    sol = solve_ivp(lambda _, y: model.rhs_(y), y0=y0, t_span=t_span)

    compartment_idx = model.compartments.index(model.to_plot)
    fig, ax = plt.subplots()
    ax.plot(sol.t, sol.y[compartment_idx, :], "k")
    ax.set_xlabel("time", fontsize=15)
    ax.set_ylabel(model.to_plot, fontsize=15)

    logger.debug(f"Model equations were integrated upto time {sol.t[-1]}.")
    logger.debug(f"Steady state estimation from integration: {sol.y[:, -1]}")

    # solve the model equations to get the steady state
    x0 = sol.y[:, -1]
    logger.debug(f"Solve model equations using scipy.optimize.root to get steady state.")
    res = root(model.rhs_, x0=x0, method="lm")

    if res.success:
        solution = res.x
        solution_dict = {c: solution[i] for i, c in enumerate(model.compartments)}
        logger.debug(f"Steady state found: {solution_dict}")

        return solution, fig

    raise RuntimeError("Could not solve model equations to find steady state!")
