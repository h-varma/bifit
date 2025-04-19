import sys
import os

file_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "../..")))

from model_equations import Model
from bifit.logging_ import logger
from bifit.preprocessing.preprocess_data import DataPreprocessor
from bifit.estimation.initial_guess import InitialGuessGenerator
from bifit.estimation.parameter_estimator import ParameterEstimator
from bifit.postprocessing.pickler import create_folder_for_results
from bifit.postprocessing.pickler import save_results_as_pickle
import bifit.postprocessing.plot_decorator as plot_decorator


def main():
    """
    Main function to run the parameter estimation process.

    Steps:
        1. Load the model and parameters.
        2. Load and preprocess the data.
        3. Generate initial guesses for the optimization variables.
        4. Solve parameter estimation problem.
        5. Compute confidence intervals.
        6. Save the results (optional).

    If the process fails with the selected bifurcation point,
    it re-tries with a different choice.
    """

    # Load the model and its settings
    model = Model()

    # Preprocess the data
    data_preprocessor = DataPreprocessor()
    # to load real measurement data with measurement errors, use function below instead:
    # data_preprocessor.load_the_data(file_path=file_path, error_scale=0.05)
    data_preprocessor.load_the_data_and_add_noise(file_path=file_path, error_scale=0.05)
    model.data = data_preprocessor.data
    model.data_weights = data_preprocessor.weights

    # Create a folder for storing the results
    results_path = create_folder_for_results(file_path)
    plot_decorator.save_plots = True
    plot_decorator.show_plots = True
    if plot_decorator.save_plots:
        plot_decorator.save_path = results_path

    # Initialize the initial guess generator
    # Set automate_bifurcation_selection to True to automatically select a bifurcation point
    initializer = InitialGuessGenerator(automate_bifurcation_selection=True)

    try_solving = True
    while try_solving:
        try:
            # Initialize the parameters and controls of the model
            model.set_parameters()

            # Generate initial guesses for all the optimization variables
            model.data = data_preprocessor.data
            model.data_weights = data_preprocessor.weights
            initializer.generate_initial_guess(model=model)

            # Solve parameter estimation problem
            fit = ParameterEstimator(
                x0=initializer.initial_guesses,
                model=model,
                method="gauss-newton",
                max_iters=100,
                plot_iters=True,
                compute_ci=True,
            )

            if fit.result.success:
                logger.info("Successfully solved the parameter estimation problem.")
                try_solving = False

        except Exception as error_message:
            logger.info("Retrying because of the following error:")
            logger.info(repr(error_message))
            fit = None
            try:
                initializer.branches.pop(initializer.selected_branch_index)
                model.mask["auxiliary_variables"] = False
                model.mask["global_parameters"] = False

                if len(initializer.branches) == 0:
                    try_solving = False
                    logger.info("Tried all the available branches. Giving up.")
            except Exception:
                try_solving = False
                logger.info("Failed to solve the parameter estimation problem. Giving up.")

        save_results_as_pickle(model=model, res=fit, path=results_path)


if __name__ == "__main__":
    main()
