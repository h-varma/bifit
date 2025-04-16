import autograd.numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from scipy.stats import chi2
from ...optimization.single_experiment.base_optimizer import BaseOptimizer
from ...optimization.check_regularity import check_constraint_qualification
from ...optimization.check_regularity import check_positive_definiteness
from ...optimization.line_search import line_search
from ...postprocessing.plot_decorator import handle_plots
from ...logging_ import logger


class BaseMultiExperimentOptimizer(BaseOptimizer, ABC):
    """
    Interface for multi-experiment optimizers. Not functional on its own.
    """

    def __init__(self):
        """
        Initializes the optimizer.
        """
        super().__init__()

        self.f = None
        self.J = None

        self.f1 = np.array([])
        self.f2 = np.array([])
        self.lagrange_multipliers = []

    def minimize(self, line_search_strategy: str):
        """
        Minimizes the objective function subject to equality constraints.

        Args:
            line_search_strategy (str): Name of the line search strategy.

        Raises:
            RuntimeError: If an error occurs during Jacobian evaluation, solving the linearized system, or line search.
        """
        x = np.copy(self.x0)

        for i in range(self.max_iters):
            self.f = self._function_evaluation(x=x)
            try:
                self.J = self._jacobian_evaluation(x=x)
            except Exception as error_message:
                self.result.success = False
                self.result.message = f"Error in evaluating Jacobian: {repr(error_message)}"
                logger.error(self.result.message)
                raise RuntimeError(f"Error in evaluating Jacobian: {repr(error_message)}")

            if not check_positive_definiteness(self.J):
                logger.warn(f"Positive definiteness does not hold in iterate {i}!")

            try:
                dxbar = self._solve_linearized_system()
            except Exception as error_message:
                self.result.success = False
                self.result.message = f"Error in solving linearized system: {repr(error_message)}"
                logger.error(self.result.message)
                raise RuntimeError(f"Error in solving linearized system: {repr(error_message)}")

            try:
                if line_search_strategy in ["exact", "armijo-backtracking"]:
                    t = line_search(
                        x=x,
                        dx=dxbar,
                        func=lambda z: self._level_function(z)[0],
                        strategy=line_search_strategy,
                    )
                else:
                    raise ValueError("Unknown line search strategy!")

            except Exception as error_message:
                self.result.success = False
                self.result.message = f"Error in line search: {repr(error_message)}"
                logger.error(self.result.message)
                raise RuntimeError(f"Error in line search: {repr(error_message)}")

            dx = dxbar * t
            x = x + dx

            self.result.x = x
            level_function, alpha = self._level_function(x=x)
            self.alpha.append(alpha)
            self.result.func = level_function
            self.result.level_functions.append(level_function)
            self.result.n_iters = i + 1

            if np.linalg.norm(dxbar) < self.xtol or level_function < self.ftol:
                self.result.success = True
                if level_function > self.ftol:
                    self.result.message = "Parameter estimation converged, but level function value is above tolerance!"
                self.result.message = "Parameter estimation solver converged!"
                logger.info(self.result.message)
                break

            if i == self.max_iters - 1:
                self.result.message = "Maximum number of iterations reached!"
                logger.error(self.result.message)
                raise RuntimeError(self.result.message)

        if self.plot_iters:
            self._plot_iterations()

        if self.compute_ci and self.result.success:
            self.result.covariance_matrix = self._compute_covariance_matrix()
            self.result.confidence_intervals = self._compute_confidence_intervals()

    @abstractmethod
    def _solve_linearized_system(self) -> np.ndarray:
        """
        Solve the linearized system.

        Returns
        -------
        np.ndarray : solution vector
        """
        raise NotImplementedError

    @abstractmethod
    def _compute_covariance_matrix(self) -> np.ndarray:
        """
        Compute the covariance matrix.

        Returns
        -------
        np.ndarray : covariance matrix
        """
        raise NotImplementedError

    def _compute_confidence_intervals(self, significance: float = 0.05) -> np.ndarray:
        """
        Computes confidence intervals for the solution vector.

        Args:
            significance (float): Significance level of the confidence intervals.

        Returns:
            np.ndarray: Confidence intervals.
        """
        C = self.result.covariance_matrix
        Cii = np.diag(C)

        self._function_evaluation(self.result.x)

        n_constraints = len(self.f2)
        mbar = self.n_total_parameters - n_constraints

        # compute quantile of Fisher distribution
        X = chi2.ppf(significance, mbar)
        return np.sqrt(X * Cii)

    @handle_plots(plot_name="fitting_iterations")
    def _plot_iterations(self):
        """
        Plots the level function at each iteration.
        """
        iterations = np.arange(0, len(self.result.level_functions))
        function_values = self.result.level_functions

        fig, ax = plt.subplots()
        ax.plot(iterations, function_values, marker="o", color="black")
        ax.set_xlabel("number of iterations")
        ax.set_ylabel("level function value")
        return None, fig

    def split_into_experiments(self, x: np.ndarray) -> np.ndarray:
        """
        Splits the solution vector into local experiments with common global parameters.

        Args:
            x (np.ndarray): Solution vector.

        Returns:
            np.ndarray: Solution matrix with experiments as rows.
        """
        local_x = x[: -self.n_global]
        local_x = local_x.reshape(self.n_experiments, self.n_local)

        global_x = x[-self.n_global :]
        global_x = global_x.reshape(-1, 1)
        global_x = np.tile(global_x, self.n_experiments)

        return np.column_stack((local_x, global_x.T))

    def _function_evaluation(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the objective and constraints.

        Args:
            x (np.ndarray): Solution vector.

        Returns:
            np.ndarray: Function values.
        """
        f = np.array([])

        self.f1 = self.f1_fun(x)
        self.f2 = np.array([])

        x = self.split_into_experiments(x)
        for i in range(self.n_experiments):
            f1 = self.f1[i * self.n_observables : (i + 1) * self.n_observables]
            f2 = self.f2_fun(x[i])
            f = np.concatenate((f, f2, f1))
            self.f2 = np.concatenate((self.f2, f2))

        return f

    def _jacobian_evaluation(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the Jacobian of the objective and constraints.

        Args:
            x (np.ndarray): Solution vector.

        Returns:
            np.ndarray: Jacobian matrix.

        Raises:
            AssertionError: If constraint qualification does not hold.
        """
        n_cols = self.n_total_parameters
        J = np.array([]).reshape(0, n_cols)

        self.j1 = self.j1_fun(x)
        self.j2 = np.array([]).reshape(0, n_cols)

        x = self.split_into_experiments(x)
        for i in range(self.n_experiments):
            j1 = self.j1[i * self.n_observables : (i + 1) * self.n_observables]

            n_rows = self.j2_fun(x[i]).shape[0]
            j2 = np.zeros((n_rows, n_cols))
            j2_ = self.j2_fun(x[i])
            local_idx = slice(i * self.n_local, (i + 1) * self.n_local)
            global_idx = slice(self.n_experiments * self.n_local, None)
            j2[:, local_idx] = j2_[:, : self.n_local]
            j2[:, global_idx] = j2_[:, self.n_local :]
            assert check_constraint_qualification(
                j2
            ), f"Experiment {i}: No constraint qualification!"

            J = np.row_stack((J, j2, j1))
            self.j2 = np.row_stack((self.j2, j2))
        assert check_constraint_qualification(self.j2), "Constraint qualification does not hold!"

        return J

    def _level_function(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Computes the value of the level function.

        Args:
            x (np.ndarray): Value of the current iterate.

        Returns:
            tuple[float, np.ndarray]: Value of the level function and constraint weights.
        """
        function_value = 0.5 * np.linalg.norm(self.f1_fun(x), ord=2) ** 2

        a = self.alpha[-1].reshape(self.n_experiments, -1)
        new_lagrange = np.abs(self.lagrange_multipliers[-1].reshape(self.n_experiments, -1))

        alpha_list = []
        for i in range(self.n_experiments):
            y = np.hstack((x[i * self.n_local : (i + 1) * self.n_local], x[-self.n_global :]))
            f2 = np.abs(self.f2_fun(y))
            alpha = np.maximum(new_lagrange[i, :], (a[i, :] + new_lagrange[i, :]) / 2)
            alpha_list.append(alpha)
            function_value += np.sum(alpha * f2)

        return function_value, np.hstack(alpha_list)
