import autograd.numpy as np
from dataclasses import dataclass


@dataclass
class ProblemSpecifications:
    """
    Represents the specifications of a problem for a mathematical model.

    Attributes:
        name (str): Name of the problem.
        compartments (list[str]): List of compartments in the model.
        to_plot (str): Variable to plot.
        true_parameters (dict[str, float]): True parameter values.
        initial_parameters (dict[str, float]): Initial guesses for parameters.
        parameters (dict[str, dict[str, float]]): Parameter details.
        controls (dict[str, str]): Control variables.
        global_parameters (list[str]): List of global parameters.
        initial_state (np.ndarray): Initial state of the model.
        integration_interval (list[float]): Time interval for integration.
        bifurcation_type (str): Type of bifurcation (e.g., "saddle-node", "hopf").
        continuation_settings (dict[str, float]): Settings for continuation analysis.
        two_parameter_continuation_method (str): Method for two-parameter continuation.
        measurement_error (str): Type of measurement error.
        exclude_states (list[np.ndarray]): States to exclude from analysis.
        multi_experiment_line_search (str): Line search method for multi-experiment analysis.
    """

    name: str

    compartments: list[str]
    to_plot: str

    true_parameters: dict[str, float]
    initial_parameters: dict[str, float]

    parameters: dict[str, dict[str, float]]

    controls: dict[str, str]
    global_parameters: list[str]

    initial_state: np.ndarray
    integration_interval: list[float]

    bifurcation_type: str
    continuation_settings: dict[str, float]

    two_parameter_continuation_method: str

    measurement_error: str

    exclude_states: list[np.ndarray]

    multi_experiment_line_search: str
