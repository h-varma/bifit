import copy
from typing import Union
import autograd.numpy as np
from autograd import jacobian
import scipy
from scipy.optimize import root
from ..continuation.base_continuer import Continuer
from ..logging_ import logger

rng = np.random.default_rng(0)


class DeflatedContinuation(Continuer):
    """
    Implements the deflated continuation method.
    """

    def __init__(
        self,
        func: callable,
        x0: np.ndarray,
        p0: float = np.nan,
        p_min: float = 0,
        p_max: float = np.inf,
        p_step: float = 1,
        p_idx: int = None,
        max_failed_attempts: int = 3,
        unique_indices: np.ndarray = None,
        data: np.ndarray = None,
    ):
        """
        Initialize the deflated continuation method.

        Args:
            func (callable): Function of x and p.
            x0 (np.ndarray): Initial guess.
            p0 (float, optional): Initial value of the parameter. Defaults to np.nan.
            p_min (float, optional): Minimum value of the parameter. Defaults to 0.
            p_max (float, optional): Maximum value of the parameter. Defaults to np.inf.
            p_step (float, optional): Step size of the parameter. Defaults to 1.
            p_idx (int, optional): Index of the parameter in the input to `func`. Defaults to None.
            max_failed_attempts (int, optional): Maximum number of failed attempts with deflation. Defaults to 3.
            unique_indices (np.ndarray, optional): Indices of variables to use for deflation. Defaults to None.
            data (np.ndarray, optional): Data points to trace. Defaults to None.
        """
        super().__init__(
            func=func,
            x0=x0,
            p0=p0,
            p_min=p_min,
            p_max=p_max,
            p_step=p_step,
            p_idx=p_idx,
            data=data,
        )

        self.init_step = p_step
        self.adaptive_steps = False if data is None else True

        self.unique_indices = unique_indices
        self.max_failed_attempts = max_failed_attempts
        self.bifurcations_found = False

        self.jacobian_ = jacobian(self.func)

        if self.p_idx is None:
            self.p_idx = len(x0)

        self._parameters = None
        self._solutions = None

        self._compute_solutions(direction=1)
        forward_solutions = copy.deepcopy(self._solutions)
        forward_parameters = copy.deepcopy(self._parameters)

        self._compute_solutions(direction=-1)
        backward_solutions = copy.deepcopy(self._solutions)
        backward_parameters = copy.deepcopy(self._parameters)

        self.parameters = backward_parameters[::-1] + forward_parameters
        self.solutions = backward_solutions[::-1] + forward_solutions

        self.__remove_empty_lists()
        self.__sort_the_solutions()

    def _compute_solutions(self, direction: int):
        """
        Find solutions in the given direction.

        Args:
            direction (int): Direction of continuation (1 for forward, -1 for backward).
        """
        x = [self.x0]
        p = np.copy(self.p0)
        p_min = copy.deepcopy(self.p0) if direction == 1 else self.p_min
        p_max = copy.deepcopy(self.p0) if direction == -1 else self.p_max

        self.p_step = copy.deepcopy(self.init_step)
        self._parameters = [p]
        self._solutions = [[]]

        solutions = copy.deepcopy(x)

        while p_min <= p <= p_max:

            # look for new (disconnected) branches
            logger.debug(f"Deflating at parameter value: {p}.")
            for solution in solutions:
                failed_attempts = 0
                success = True
                while success or failed_attempts < self.max_failed_attempts:
                    x0 = np.copy(solution)
                    if len(self._solutions[-1]) > 0:
                        x0 = self.__add_noise(solution)
                    sol = self._deflation_step(x0=x0, p=p)
                    if sol is None:
                        success = False
                        failed_attempts += 1
                    else:
                        success = True
                        self._solutions[-1].append(sol)

            if len(self._solutions[-1]) == 0:
                solutions = copy.deepcopy(x)
                count_ = 0
            else:
                solutions = copy.deepcopy(self._solutions[-1])
                count_ = len(solutions)
            logger.debug(f"Found {count_} solutions at parameter value: {p}.")

            self._solutions.append([])

            # step size adaptation
            if self.data is not None:
                if self.p_step < 0.5 * self.init_step:
                    self.p_step = 2 * self.p_step
                p_step = direction * self.p_step
                self.p_step = self._trace_data(x=p, dx=1.0, step=p_step, direction=direction)
                self.p_step = np.abs(self.p_step)

            # continue existing branches
            for solution in solutions:
                sol = self._continuation_step(x0=solution, p0=p, direction=direction)
                if sol is not None:
                    self._solutions[-1].append(sol)

            p = p + direction * self.p_step
            self._parameters.append(p)

            if len(self._solutions[-1]):
                solutions = copy.deepcopy(self._solutions[-1])
                count_ = len(solutions)
                logger.debug(f"Continued {count_} solutions to parameter value: {p}.")

    def _continuation_step(self, x0: np.ndarray, p0: float, direction: int):
        """
        Perform a continuation step using a simple predictor-corrector method.

        Args:
            x0 (np.ndarray): Current solution.
            p0 (float): Current parameter value.
            direction (int): Continuation direction (1 for forward, -1 for backward).

        Returns:
            np.ndarray: New solution.
        """

        # predictor step
        x = None
        p_step = np.copy(self.p_step)

        final_p = p0 + direction * self.p_step
        while x is None or p0 + direction * p_step <= final_p:
            sol = self._join_x_and_p(x0, p0)
            Jx, Jp = self._compute_jacobians(sol)

            step_vector = self._solve_linear_system(Jx, -Jp)
            x1 = x0 + direction * p_step * step_vector
            p1 = p0 + direction * p_step

            # corrector step
            def corrector(_x):
                _sol = self._join_x_and_p(_x, p1)
                return self.func(_sol)

            res = root(corrector, x0=x1)
            x = res.x if res.success else None

            if x is None:
                p_step = p_step / 2.0
                if p_step < 0.01 * self.p_step:
                    break
            else:
                if p1 == final_p:
                    break
                x0 = x
                p0 = p1

        if x is None:
            return None

        is_in_history = self.__check_if_solution_already_exists(x)
        return None if is_in_history else x

    def _deflation_step(self, x0: np.ndarray, p: float) -> np.ndarray:
        """
        Perform a deflation step to find a new solution.

        Args:
            x0 (np.ndarray): Initial guess for the solution.
            p (float): Parameter value.

        Returns:
            np.ndarray: New solution at parameter value p.
        """
        known_solutions = self._solutions[-1]

        def _deflated_corrector(_x):
            sol = self._join_x_and_p(_x, p)
            df = self.func(sol)
            for solution in known_solutions:
                df = np.dot(self.__deflation_operator(_x, solution), df)
            return df

        res = root(_deflated_corrector, x0=x0)
        x = res.x if res.success else None

        if x is None:
            return None

        is_in_history = self.__check_if_solution_already_exists(x)
        is_valid = self._check_if_solution_satisfies_ftol(x=x, p=p, ftol=1e-4)
        return x if is_valid and not is_in_history else None

    def __deflation_operator(self, u: np.ndarray, ustar: np.ndarray) -> np.ndarray:
        """
        Operator to deflate out known solutions.

        Args:
            u (np.ndarray): Current solution.
            ustar (np.ndarray): Known solution.

        Returns:
            np.ndarray: Deflation operator value.
        """
        size_ = len(u)
        if self.unique_indices is not None:
            u = u[self.unique_indices]
            ustar = ustar[self.unique_indices]
            factor_ = 1 + (1 / np.sum((u - ustar) ** 2))
            operator_ = np.eye(size_)
            operator_[self.unique_indices, self.unique_indices] *= factor_
            return operator_
        else:
            return (1 + (1 / np.sum((u - ustar) ** 2))) * np.eye(len(u))

    @staticmethod
    def __add_noise(x: np.ndarray) -> np.ndarray:
        """
        Add normally distributed noise to the input data.

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Input data with added noise.
        """
        mean = x
        stddev = np.maximum(1, np.abs(x))

        a, b = -mean / stddev, np.inf
        return scipy.stats.truncnorm.rvs(a=a, b=b, loc=mean, scale=stddev)

    def __check_if_solution_already_exists(self, x: np.ndarray) -> bool:
        """
        Check if `x` is the same as a previously found result, differing only in their signs.

        Args:
            x (np.ndarray): Solution to check.

        Returns:
            bool: True if the solution already exists, False otherwise.
        """
        flag = [np.allclose(np.abs(x), np.abs(s), atol=0) for s in self._solutions[-1]]
        return True if True in flag else False

    def __sort_the_solutions(self):
        """
        Sort steady states at a parameter value to continuously follow
        steady states from the previous parameter value.
        """
        max_size = max([len(sol) for sol in self.solutions])
        for i in range(len(self.solutions)):
            self.solutions[i] = np.row_stack(self.solutions[i])
            if self.solutions[i].shape[0] < max_size:
                idx = np.arange(self.solutions[i].shape[0], max_size)
                for j in idx:
                    self.solutions[i] = self.__insert_nan_rows(self.solutions[i], j)

        assert all([sol.shape[0] == max_size for sol in self.solutions])

    @staticmethod
    def __insert_nan_rows(x: np.ndarray, idx: Union[int, list, slice]) -> np.ndarray:
        """
        Insert rows filled with nan into the input matrix.

        Args:
            x (np.ndarray): Input matrix.
            idx (Union[int, list, slice]): Index at which to insert nan rows.

        Returns:
            np.ndarray: Matrix with nan rows inserted.
        """
        return np.insert(x, idx, np.nan, axis=0)

    @staticmethod
    def __size(X: np.ndarray) -> int:
        """
        Compute the number of non-nan rows in a matrix.

        Args:
            X (np.ndarray): Input matrix.

        Returns:
            int: Number of non-nan rows.
        """
        return len([X[i, :] for i in range(X.shape[0]) if not any(np.isnan(X[i, :]))])

    def __remove_empty_lists(self):
        """
        Remove empty solutions lists from the solutions and parameters.
        """
        idx = []
        for i, sol in enumerate(self.solutions):
            if len(sol) == 0:
                idx.append(i)
        self.parameters = [p for i, p in enumerate(self.parameters) if i not in idx]
        self.solutions = [s for i, s in enumerate(self.solutions) if i not in idx]

    def detect_saddle_node_bifurcation(self, parameter: str) -> np.ndarray:
        """
        Detect saddle-node bifurcation branches in the solutions.

        Args:
            parameter (str): Name of bifurcation parameter.

        Returns:
            np.ndarray: Saddle-node bifurcation point.
        """

        branches = [[]]
        old = self._join_x_and_p(x=self.solutions[0], p=self.parameters[0])
        for i in range(1, len(self.solutions)):
            new = self._join_x_and_p(x=self.solutions[i], p=self.parameters[i])
            change_in_solutions = self.__size(new) - self.__size(old)

            if 1 <= change_in_solutions <= 2:
                for j in range(new.shape[0]):
                    if np.isnan(old[j][:-1]).any() and not np.isnan(new[j][:-1]).any():
                        branches[-1].append(new[j])

            elif -2 <= change_in_solutions <= -1:
                for j in range(old.shape[0]):
                    if np.isnan(new[j][:-1]).any() and not np.isnan(old[j][:-1]).any():
                        branches[-1].append(old[j])

            if len(branches[-1]) == 2:
                branches[-1] = [sum(branches[-1]) / 2]
                self.bifurcations_found = True
                branches.append([])

            old = new

        branches = [branch[0] for branch in branches if len(branch) > 0]
        if len(branches):
            self.bifurcations_found = True
        return self.__select_bifurcation_point(branches=branches, parameter=parameter)

    def detect_hopf_bifurcation(self, parameter: str) -> np.ndarray:
        """
        Detect Hopf bifurcation points in the solutions.

        Args:
            parameter (str): Name of bifurcation parameter.

        Returns:
            np.ndarray: Hopf bifurcation point.
        """
        old_eigvals = self.__get_eigenvalues(self.solutions[0], self.parameters[0])
        old_signs = np.sign(old_eigvals.real)

        branches = [[]]
        for i in range(1, len(self.solutions)):
            new_eigvals = self.__get_eigenvalues(self.solutions[i], self.parameters[i])
            new_signs = np.sign(new_eigvals.real)

            if (old_signs != new_signs).any():
                sign_change = np.abs(old_signs - new_signs) == 2
                is_complex = np.iscomplex(old_eigvals) | np.iscomplex(new_eigvals)
                mask = np.any(sign_change & is_complex, axis=1)
                solution = self.solutions[i][mask, :]
                for sol in solution:
                    sol = self._join_x_and_p(sol, self.parameters[i])
                    branches[-1].extend(sol)
                    self.bifurcations_found = True
                    branches.append([])

            old_eigvals = new_eigvals
            old_signs = new_signs

        return self.__select_bifurcation_point(branches=branches, parameter=parameter)

    def __get_eigenvalues(self, solutions: np.ndarray, parameter: float) -> np.ndarray:
        """
        Compute the eigenvalues of the Jacobian matrix at each solution in solutions.

        Args:
            solutions (np.ndarray): Solutions at which to compute the eigenvalues.
            parameter (float): Parameter value.

        Returns:
            np.ndarray: Eigenvalues of the Jacobian matrix.
        """
        eigenvalues = []
        solutions = self._join_x_and_p(x=solutions, p=parameter)
        n_solutions, n_variables = solutions.shape
        for i in range(n_solutions):
            if np.any(np.isnan(solutions[i, :])):
                eigenvalues.append(np.nan * np.ones(n_variables - 1))
            else:
                jacobian_ = self.jacobian_(solutions[i, :])
                jacobian_ = np.delete(jacobian_, self.p_idx, axis=1)
                eigenvalues.append(np.linalg.eigvals(jacobian_))
        return np.row_stack([eigenvalues])

    def __select_bifurcation_point(self, branches: list, parameter: str) -> Union[np.ndarray, list]:
        """
        Select one bifurcation point from the detected bifurcation points.

        Args:
            branches (list): Bifurcation branches.
            parameter (str): Parameter name.

        Returns:
            Union[np.ndarray, list]: Bifurcation points.
        """
        branches = [branch for branch in branches if len(branch) > 0]

        if not self.bifurcations_found:
            logger.error("No bifurcations could be found in the given parameter range!")
            raise ValueError(
                "No bifurcations could be found in the given parameter range. "
                "Try using different initial guesses or expand the parameter range."
            )
        elif len(branches) == 1:
            logger.info(f"A bifurcation was detected near {parameter} = {branches[0][self.p_idx]}.")
            return branches
        else:
            logger.info(f"Bifurcations were detected near the following values of {parameter}:")
            for i, branch in enumerate(branches):
                logger.info(f"{i + 1}: {branch[self.p_idx]}")
            return branches
