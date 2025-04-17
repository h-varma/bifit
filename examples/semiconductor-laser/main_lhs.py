import sys
import os

file_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "../..")))

from model_equations import Model
from bifit.logging_ import logger
from bifit.preprocessing.sampler import generate_samples_using_lhs
from bifit.preprocessing.sampler import generate_samples_using_gaussian
from bifit.preprocessing.preprocess_data import DataPreprocessor
from bifit.parameter_estimation.initial_guess import InitialGuessGenerator
from bifit.parameter_estimation.parameter_estimator import ParameterEstimator
from bifit.postprocessing.pickler import create_folder_for_results
from bifit.postprocessing.pickler import save_results_as_pickle
import bifit.postprocessing.plot_decorator as plot_decorator


def main():
    """
    Main function to run the parameter estimation process.

    Steps:
        1. Load the model and randomize parameters.
        2. Generate samples using LHS or Gaussian sampling.
        3. Preprocess the data and add noise.
        4. Perform parameter estimation for each sample.
        5. Save the results.

    Raises:
        ValueError: If an unknown sampling method is specified.
    """
    # Load the model and its settings
    model = Model()

    # Create random initial guesses for the global parameters
    number_of_parameter_guesses = 25
    true_parameters = model.true_parameters
    to_vary = model.global_parameters

    sampling_strategy = "lhs"

    if sampling_strategy == "lhs":
        # Generate samples using latin hypercube sampling
        bounds = {"alpha": [1, 10], "B": [0.01, 0.1]}
        samples = generate_samples_using_lhs(
            parameters=true_parameters, bounds=bounds, n_points=number_of_parameter_guesses
        )
    elif sampling_strategy == "gaussian":
        # Generate samples using Gaussian distribution
        samples = generate_samples_using_gaussian(
            parameters=true_parameters,
            to_vary=to_vary,
            noise=1,
            n_points=number_of_parameter_guesses,
        )
    else:
        raise ValueError("Unknown sampling method.")

    # Preprocess the data
    data_preprocessor = DataPreprocessor()
    # to load real measurement data with measurement errors, use function below instead:
    # data_preprocessor.load_the_data(file_path=file_path, error_scale=0.05)
    data_preprocessor.load_the_data_and_add_noise(file_path=file_path, error_scale=0.05)
    model.data = data_preprocessor.data
    model.data_weights = data_preprocessor.weights
    data_entry = model.data[9]

    for i in range(samples.shape[0]):
        # Create a folder for storing the results
        results_path = create_folder_for_results(file_path)
        plot_decorator.save_plots = True
        plot_decorator.show_plots = False
        if plot_decorator.save_plots:
            plot_decorator.save_path = results_path

        # Set the parameters of the model
        model.set_parameters(controls=data_entry, parameters=samples[i, :])

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

        save_results_as_pickle(model=model, res=fit, path=results_path)


if __name__ == "__main__":
    main()
