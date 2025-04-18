import autograd.numpy as np
from ..initial_guess.steady_state import find_steady_state
from ..initial_guess.steady_state_curve import compute_steady_state_curve
from ..initial_guess.bifurcation_point import get_bifurcation_point
from ..initial_guess.trace_data import trace_measured_bifurcations
from ..initial_guess.match_solutions import match_solutions_to_data
from ..models.utils import nparray_to_dict
from ..logging_ import logger


class InitialGuessGenerator:
    """
    Generates initial guesses for the parameter estimation problem.
    """

    def __init__(self):
        """
        Initializes the initial guess generator.
        """
        self.model = None
        self.branches = []

    def generate_initial_guess(self, model: object):
        """
        Generates initial guesses for the parameter estimation problem.

        Args:
            model (object): Details of the model.

        Raises:
            RuntimeError: If initial guess generation fails.
        """
        self.model = model
        try:
            if len(self.branches) == 0:
                # Step 1: Get a steady state solution
                logger.info("Step 1: Find a steady state solution.")
                self.steady_state, _ = find_steady_state(model=self.model)

                # Step 2: Draw a bifurcation diagram and select a bifurcation point for continuation
                logger.info("Step 2: Continue the steady state to draw a bifurcation diagram.")
                kwargs = {
                    "model": self.model,
                    "continuer_name": "deflated",
                }
                self.branches, _ = compute_steady_state_curve(x0=self.steady_state, **kwargs)

            # Step 3: Get the exact bifurcation point from the approximation
            logger.info("Step 3: Get the exact bifurcation point from the approximation.")
            kwargs = {"model": self.model, "optimizer_name": "scipy"}
            self.bifurcation_point = get_bifurcation_point(branch=self.branches[0], **kwargs)

            # Step 4: Continue the bifurcation point to trace the data
            logger.info("Step 4: Trace a two-parameter bifurcation diagram along the data.")
            kwargs = {
                "model": self.model,
                "continuer_name": model.two_parameter_continuation_method,
            }
            self.bifurcation_points, _ = trace_measured_bifurcations(
                x0=self.bifurcation_point, **kwargs
            )

            # Step 5: Set up the initial guesses
            logger.info("Step 5: Match the predicted points to experimental data.")
            self.initial_guesses = match_solutions_to_data(
                model=self.model, solutions=self.bifurcation_points, fill_missing=False
            )

            # Step 6: Append global parameters to the initial guess
            logger.info("Step 6: Append global parameters to the initial guess.")
            for global_parameter in self.model.global_parameters:
                self.model.parameters[global_parameter]["vary"] = True
                value = self.model.parameters[global_parameter]["value"]
                self.initial_guesses = np.hstack([self.initial_guesses, value])
            self.model.mask["global_parameters"] = True
        except Exception as error_message:
            self.initial_guesses = None
            self.model.error_message = repr(error_message)
            logger.error(f"Initial guess generation failed: {repr(error_message)}")
            raise RuntimeError(f"Initial guess generation failed: {repr(error_message)}")
