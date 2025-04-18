import autograd.numpy as np
import matplotlib.pyplot as plt
from matplotlib import container, lines
from ..models.utils import nparray_to_dict
from ..estimation.problem_generator import OptimizationProblemGenerator
from ..optimization.multi_experiment.select_optimizer import import_optimizer
from ..postprocessing.plot_decorator import handle_plots
from ..logging_ import logger


class ParameterEstimator:
    """
    Solves the multi-experiment parameter estimation problem.
    """

    def __init__(
        self,
        x0: np.ndarray,
        model: object,
        method: str = "osqp",
        xtol: float = 1e-4,
        ftol: float = 1e-4,
        max_iters: int = 100,
        plot_iters: bool = False,
        compute_ci: bool = False,
    ):
        """
        Initializes the parameter estimator.

        Args:
            x0 (np.ndarray): Initial guess.
            model (object): Details of the model.
            method (str): Approach to solve the problem. Defaults to "osqp".
            xtol (float): Convergence threshold for step size. Defaults to 1e-4.
            ftol (float): Convergence threshold for function value. Defaults to 1e-4.
            max_iters (int): Maximum number of iterations. Defaults to 100.
            plot_iters (bool): Whether to plot the level function at each iteration. Defaults to False.
            compute_ci (bool): Whether to compute confidence intervals. Defaults to False.
        """
        if x0 is None:
            return None

        self.x0 = x0
        self.model = model
        self.xtol = xtol
        self.ftol = ftol
        self.max_iters = max_iters
        self.plot_iters = plot_iters
        self.compute_ci = compute_ci
        self.method = method

        self.n_experiments = len(model.data)
        self.n_observables = len(model.controls)
        self.n_global = len(model.global_parameters)
        self.n_local = len(model.compartments) + len(model.controls)
        if model.bifurcation_type == "saddle-node":
            self.n_local += len(model.compartments)
        elif model.bifurcation_type == "hopf":
            self.n_local += 2 * len(model.compartments) + 1
        else:
            raise Exception("Invalid bifurcation type!")

        self.problem = OptimizationProblemGenerator(
            model=model,
            include_steady_state=True,
            include_singularity=True,
            include_normalization=True,
        )
        self.equality_constraints = self.problem.stack_functions

        self.Solver = import_optimizer(method)

        logger.info(f"Estimate the model parameters using {method} solver.")
        self.result = self.__run_solver()

        if self.result.success:
            logger.info(f"Solver has converged in {self.result.n_iters} iterations!")
            logger.info(f"Initial guesses: {self.__get_global_parameters(self.x0)}.")
            logger.info(f"Solutions: {self.__get_global_parameters(self.result.x)}.")
            if compute_ci:
                CI = self.__get_global_parameters(self.result.confidence_intervals)
                logger.info(f"Confidence intervals: {CI}.")
            self.__plot_results()

    def __run_solver(self):
        """
        Runs the optimization solver.

        Returns:
            object: Solver result object.
        """
        self.solver = self.Solver(
            x0=self.x0,
            f1_fun=self.objective_function,
            f2_fun=self.equality_constraints,
            n_local=self.n_local,
            n_global=self.n_global,
            n_observables=self.n_observables,
            n_experiments=self.n_experiments,
            xtol=self.xtol,
            ftol=self.ftol,
            max_iters=self.max_iters,
            plot_iters=self.plot_iters,
            compute_ci=self.compute_ci,
        )
        line_search = self.model.multi_experiment_line_search
        self.solver.minimize(line_search_strategy=line_search)
        return self.solver.result

    def __get_global_parameters(self, x: np.ndarray) -> dict:
        """
        Extracts the global parameters from the multi-experiment solution vector.

        Args:
            x (np.ndarray): Solution vector.

        Returns:
            dict: Global parameters.
        """
        parameters = dict()
        for i, key in enumerate(self.model.global_parameters):
            parameters[key] = x[-self.n_global + i]
        return parameters

    @handle_plots(plot_name="fitting_results")
    def __plot_results(self):
        """
        Plots the results of the parameter estimation.
        """
        h_param = self.model.controls["homotopy"]
        f_param = self.model.controls["free"]

        h_data = np.array([d[h_param] for d in self.model.data])
        f_data = np.array([d[f_param] for d in self.model.data])

        solutions = self.solver.split_into_experiments(self.result.x)
        initial_guesses = self.solver.split_into_experiments(self.x0)

        fig, ax = plt.subplots()
        ax.plot(h_data, f_data, "X", color="black", label="measurements")
        for i in range(self.n_experiments):
            _, p0, _ = nparray_to_dict(initial_guesses[i], model=self.model)
            label = "initial guess" if i == 0 else None
            ax.plot(p0[h_param], p0[f_param], "o", color="#007EE3", alpha=0.2, label=label)

            label = "solution" if i == 0 else None
            _, p, _ = nparray_to_dict(solutions[i], model=self.model)
            if self.compute_ci:
                CI = self.result.confidence_intervals
                error = self.solver.split_into_experiments(CI)
                _, perr, _ = nparray_to_dict(error[i], model=self.model)
                _, _, bars = ax.errorbar(
                    x=p[h_param],
                    y=p[f_param],
                    xerr=perr[h_param],
                    yerr=perr[f_param],
                    ecolor="black",
                    marker="o",
                    mfc="#007EE3",
                    mec="#007EE3",
                    label=label,
                )
                [bar.set_alpha(0.4) for bar in bars]

                # get handles
                handles, labels = ax.get_legend_handles_labels()
                # remove the errorbars
                for k, _h in enumerate(handles):
                    if isinstance(_h, container.ErrorbarContainer):
                        handles[k] = lines.Line2D(
                            [], [], linestyle="None", marker="o", mfc="#007EE3", mec="#007EE3"
                        )
                ax.legend(handles, labels)
            else:
                ax.plot(p[h_param], p[f_param], "o", color="#007EE3", label=label)
                ax.legend()

        ax.set_xlabel(h_param, fontsize=15)
        ax.set_ylabel(f_param, fontsize=15)
        return _, fig

    def objective_function(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the objective function for the parameter estimation problem.

        Args:
            x (np.ndarray): Solution vector.

        Returns:
            np.ndarray: Residuals.
        """
        obj_fun = np.array([])
        global_x = x[-self.n_global :]
        for i, (data, weights) in enumerate(zip(self.model.data, self.model.data_weights)):
            local_x = x[i * self.n_local : (i + 1) * self.n_local]
            solution = np.concatenate((local_x, global_x))
            _, p, _ = nparray_to_dict(solution, model=self.model)

            for key in self.model.controls.values():
                if self.model.measurement_error == "absolute_linear":
                    residual = (p[key] - data[key]) / weights[key]
                else:
                    raise Exception("Invalid measurement error type!")

                obj_fun = np.hstack((obj_fun, residual))

        return obj_fun
