import sys
import os

file_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "../..")))

import copy
import easygui
import pickle
import numpy as np
from model_equations import Model
from bifit.logging_ import logger
from bifit.estimation.initial_guess import InitialGuessGenerator
from bifit.estimation.parameter_estimator import ParameterEstimator
import bifit.postprocessing.plot_decorator as plot_decorator


def main():
    """
    Main function to load results and perform parameter estimation.

    Steps:
        1. Load the model and randomize parameters.
        2. Load results from a selected directory.
        3. Extract data and weights.
        4. Perform parameter estimation using the loaded results.

    Raises:
        FileNotFoundError: If no results are found in the selected directory.
    """
    # Load the model and randomize the parameters
    model = Model()

    # Load the results
    current_path = os.path.dirname(__file__)
    sys.path.append(os.path.abspath(os.path.join(current_path, "../..")))
    folder_path = easygui.diropenbox()

    try:
        with open(os.path.join(folder_path, "summary.pkl"), "rb") as f:
            results = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("No results found in the selected directory.")

    # Extract the data
    data_values = results["data"]["values"]
    data_weights = copy.deepcopy(data_values)
    for i, data in enumerate(data_values):
        for key, value in data.items():
            data_weights[i][key] = np.abs(value * results["data"]["noise"])

    plot_decorator.save_plots = False
    plot_decorator.show_plots = True

    # Set the parameters of the model
    model.parameters = {}
    for p_name, p_value in results["model"]["parameters"].items():
        model.parameters[p_name] = {"value": p_value, "vary": False}

    model.mask = {
        "compartments": False,
        "controls": False,
        "auxiliary_variables": False,
        "global_parameters": False,
    }

    initializer = InitialGuessGenerator()

    try_solving = True
    while try_solving:
        try:
            # Generate initial guesses for the parameter estimation
            model.data = data_values
            model.data_weights = data_weights
            initializer.generate_initial_guess(model=model)

            # Solve parameter estimation problem
            fit = ParameterEstimator(
                x0=initializer.initial_guesses,
                model=model,
                method="gauss-newton",
                max_iters=100,
                plot_iters=True,
                compute_ci=True,
                xtol=1e-3,
                ftol=1e-3,
            )

            if fit.result.success:
                logger.info("Successfully solved the parameter estimation problem.")
                try_solving = False

        except Exception as error_message:
            logger.info("Retrying because of the following error:")
            logger.info(repr(error_message))
            fit = None
            try:
                initializer.branches.pop(0)
                model.mask["auxiliary_variables"] = False
                model.mask["global_parameters"] = False
                for parameter_name, parameter_value in model.true_parameters.items():
                    if parameter_name in model.global_parameters + [model.controls["free"]]:
                        model.parameters[parameter_name]["vary"] = False
                if len(initializer.branches) == 0:
                    try_solving = False
                    logger.info("Tried all the available branches. Giving up.")
            except Exception:
                try_solving = False
                logger.info("Failed to solve the parameter estimation problem. Giving up.")


if __name__ == "__main__":
    main()
