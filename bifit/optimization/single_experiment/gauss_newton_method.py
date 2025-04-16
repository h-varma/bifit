import copy
from typing import Union

import autograd.numpy as np
from autograd import jacobian
from scipy.optimize import minimize

from ...optimization.check_regularity import check_constraint_qualification
from ...optimization.line_search import line_search
from ...optimization.single_experiment.base_optimizer import OptimizerResult


class GeneralizedGaussNewton:
    """
    Generalized Gauss-Newton method for constrained optimization.

    Args:
        objective (callable): Objective function.
        constraints (Union[list, dict]): Dictionary containing the equality and inequality constraints.
        x0 (np.ndarray): Initial guess.
        tol (float, optional): Tolerance for convergence. Defaults to 1e-8.
        max_iters (int, optional): Maximum number of iterations. Defaults to 100.
        active_iters (int, optional): Maximum iterations for active set strategy. Defaults to 50.
        zero (float, optional): Threshold for zero. Defaults to 1e-6.
    """

    def __init__(
        self,
        objective: callable,
        constraints: Union[list, dict],
        x0: np.ndarray,
        tol: float = 1e-8,
        max_iters: int = 100,
        active_iters: int = 50,
        zero: float = 1e-6,
    ):

        self.f1_func = objective
        self.f2_func = constraints["equality"]
        self.f3_func = constraints["inequality"]

        if self.f2_func(x0) is None:
            self.f2_func = None
        if self.f3_func(x0) is None:
            self.f3_func = None

        self.j1_func = jacobian(self.f1_func)
        self.j2_func = jacobian(self.f2_func) if self.f2_func is not None else None
        self.j3_func = jacobian(self.f3_func) if self.f3_func is not None else None

        if self.f3_func is None:
            self.solver = self.__compute_step_direction
        else:
            self.solver = self.active_set_strategy

        self.result = OptimizerResult()
        self.adjoint_variables = None

        self.x0 = x0
        self.tol = tol
        self.max_iters = max_iters
        self.active_iters = active_iters
        self.zero = zero
        self.minimize()

    def minimize(self):
        """
        Solve the optimization problem.

        Returns:
            None
        """
        sol = dict()
        sol["x"] = np.copy(self.x0)
        for i in range(self.max_iters):
            sol["f1"] = self.f1_func(sol["x"])
            sol["f2"] = np.array([]) if self.f2_func is None else self.f2_func(sol["x"])
            sol["f3"] = np.array([]) if self.f3_func is None else self.f3_func(sol["x"])
            sol["j1"] = self.j1_func(sol["x"])
            sol["j2"] = np.array([]) if self.j2_func is None else self.j2_func(sol["x"])
            sol["j3"] = np.array([]) if self.j3_func is None else self.j3_func(sol["x"])

            dx = self.solver(sol=sol)
            if self.f2_func is not None:
                self.compute_adjoint_variables(dx=dx, sol=sol)

            t = line_search(
                x=sol["x"],
                dx=dx,
                func=self.level_function,
                strategy="armijo-backtracking",
            )

            sol["x"] = sol["x"] + dx * t

            self.result.x = sol["x"]
            self.result.func = self.level_function(x=sol["x"])
            self.result.n_iters = i + 1

            if np.linalg.norm(dx * t) < self.tol:
                self.result.success = True
                self.result.message = "Minimizer converged successfully."
                break
        if self.result.n_iters == self.max_iters:
            self.result.message = "Maximum number of iterations reached."

    def check_feasibility(self, dy: np.ndarray, sol: dict) -> float:
        """
        Measure feasibility of the linearization at the current solution.

        Args:
            dy (np.ndarray): Step direction.
            sol (dict): Dictionary containing the current solution.

        Returns:
            float: Constraint violation.
        """
        constraint_violation = 0
        if self.f2_func is not None:
            f2_violation = sol["f2"] + np.dot(sol["j2"], dy)
            constraint_violation += np.sum(f2_violation**2)
        if self.f3_func is not None:
            f3_violation = -(sol["f3"] + np.dot(sol["j3"], dy))
            f3_violation = np.maximum(0, f3_violation)
            constraint_violation += np.sum(f3_violation**2)
        return constraint_violation

    def get_feasible_point(self, sol: dict) -> np.ndarray:
        """
        Get a feasible solution starting from the initial guess.

        Args:
            sol (dict): Dictionary containing the current solution.

        Returns:
            np.ndarray: Feasible solution.
        """
        x0 = np.zeros_like(sol["x"])
        res = minimize(
            self.check_feasibility,
            x0=x0,
            args=(sol,),
            method="SLSQP",
            options={"ftol": 1e-10},
        )
        assert res.success, "Active set strategy: error in finding a feasible point!"
        return res.x

    @staticmethod
    def values_of_inactive_constraints(dx: np.ndarray, sol: dict) -> np.ndarray:
        """
        Compute linearization of inactive inequality constraints.

        Args:
            dx (np.ndarray): Step direction.
            sol (dict): Dictionary containing the current solution.

        Returns:
            np.ndarray: Linearized constraints.
        """
        bbar = np.nan * np.ones_like(sol["f3"])
        linearized_constraint = sol["f3"] + np.dot(sol["j3"], dx)
        bbar[sol["inactive"]] = linearized_constraint[sol["inactive"]]
        return bbar

    def active_set_strategy(self, sol: dict) -> np.ndarray:
        """
        Active set strategy for solving inequality-constrained optimization.

        Args:
            sol (dict): Dictionary containing the current solution.

        Returns:
            np.ndarray: Step direction.
        """
        dx = self.get_feasible_point(sol=sol)

        linearized_constraint = sol["f3"] + np.dot(sol["j3"], dx)
        sol["active"] = self.where_zero(linearized_constraint, tol=self.zero)
        sol["inactive"] = self.where_positive(linearized_constraint, tol=self.zero)

        b = np.nan * np.ones_like(sol["f3"])
        b[sol["inactive"]] = linearized_constraint[sol["inactive"]]

        # solve the equality constrained problem: J0 * dx + F == 0
        dx = self.__compute_step_direction(sol=sol)

        # set a boolean to check if the solution has been found
        solution_found = False

        # if any of the previously inactive inequality constraints have now become negative ...
        k = 0  # counter to interrupt loop in case the active sets start oscillating
        while not solution_found:
            # STEP 1
            # compute the values of the previously inactive inequality constraints
            bbar = self.values_of_inactive_constraints(dx=dx, sol=sol)

            while (bbar < -self.zero).any():
                # find the indices of the previously inactive inequality constraints that have now become negative
                idx = self.where_negative(bbar, tol=self.zero)
                # add a new active inequality constraint by taking the constraint that drops the most to negative values
                # and finding its active form along the line between bbar and b[mu-1]
                a_fun = np.inf * np.ones_like(bbar)
                a_fun[idx] = b[idx] / (b[idx] - bbar[idx])
                a_idx = np.argmin(a_fun)
                a = a_fun[a_idx]
                # update b
                bmu = np.nan * np.ones_like(sol["f3"])
                interpolation = (1 - a) * b + a * bbar
                bmu[sol["inactive"]] = interpolation[sol["inactive"]]
                b = copy.deepcopy(bmu)
                # add the index where alpha was found to the set of active indices
                sol["active"] = np.sort(np.append(sol["active"], a_idx))
                sol["inactive"] = sol["inactive"][sol["inactive"] != a_idx]

                # STEP 2
                # solve the equality constrained problem: J0 * dx + F == 0 again
                dx = self.__compute_step_direction(sol=sol)

                # STEP 1 again
                # compute the values of the previously inactive inequality constraints with dx
                bbar = self.values_of_inactive_constraints(dx=dx, sol=sol)

            # STEP 3
            # compute the lagrange multipliers
            self.compute_adjoint_variables(dx=dx, sol=sol)
            if sol["active"].size != 0:
                m2 = sol["j2"].shape[0]
                linearized_constraint_multipliers = self.adjoint_variables[m2:]
                active_multipliers = linearized_constraint_multipliers[sol["active"]]
                if (active_multipliers >= -self.zero).all():
                    # solved if all the lagrange multipliers of the active inequality constraints are non-negative
                    solution_found = True
                else:
                    # remove the inequality constraint with the most negative lagrange multiplier from active set
                    lambdas = np.inf * np.ones_like(sol["f3"])
                    lambdas[sol["active"]] = active_multipliers
                    idx = np.argmin(lambdas)
                    sol["inactive"] = np.sort(np.append(sol["inactive"], idx))
                    sol["active"] = sol["active"][sol["active"] != idx]
                    # update b
                    bbar[idx] = 0
                    b = copy.deepcopy(bbar)

                    # STEP 2 again
                    # solve the equality constrained problem: J0 * dx + F == 0 again
                    dx = self.__compute_step_direction(sol=sol)

            else:
                # the active set is empty
                solution_found = True
            k += 1
            # if the active sets oscillate for more than 100 iterations, interrupt the look and return dx as is
            if k >= self.active_iters:
                return dx
        return dx

    def compute_adjoint_variables(self, dx: np.ndarray, sol: dict):
        """
        Compute adjoint variables for equality and inequality constraints.

        Args:
            dx (np.ndarray): Step direction.
            sol (dict): Dictionary containing the current solution.

        Returns:
            None
        """
        n = sol["j1"].shape[1]
        jc = np.row_stack((sol["j2"].reshape((-1, n)), sol["j3"].reshape((-1, n))))
        Q, R = np.linalg.qr(jc.T, mode="complete")

        f1_linear = np.dot(sol["j1"], dx) + sol["f1"]
        self.adjoint_variables = np.linalg.lstsq(R, Q.T @ sol["j1"].T @ f1_linear)[0]

    @staticmethod
    def __compute_step_direction(sol: dict) -> np.ndarray:
        """
        Compute step direction for equality-constrained optimization.

        Args:
            sol (dict): Dictionary containing the current solution.

        Returns:
            np.ndarray: Step direction.
        """
        n1 = sol["j1"].shape[1]
        f2 = sol["f2"] if sol["f2"].size else np.zeros((0,))
        j2 = sol["j2"] if sol["f2"].size else np.zeros((0, n1))
        f3, j3 = np.zeros((0,)), np.zeros((0, n1))
        if sol["f3"].size:
            if [sol["active"]].size:
                f3, j3 = sol["f3"][sol["active"]], sol["j3"][sol["active"]]
        fc = np.concatenate((f2, f3))
        Jc = np.row_stack((j2, j3))

        if fc.size:
            check_constraint_qualification(Jc)
            Q, R = np.linalg.qr(Jc.T, mode="complete")
            Rbar = R[: R.shape[1], : R.shape[1]]
            dy1 = -np.dot(Rbar.T, fc)

            A = sol["j1"] @ Q
            A1, A2 = A[:, : fc.shape[0]], A[:, fc.shape[0] :]

            dy2 = np.array([])
            if A2.size:
                dy2 = np.linalg.lstsq(A2, -(sol["f1"] + A1 @ dy1))[0]
            return Q @ np.concatenate((dy1, dy2))
        else:
            return np.linalg.lstsq(sol["j1"], -sol["f1"])[0]

    def level_function(self, x) -> float:
        """
        Compute the value of the level function.

        Args:
            x (np.ndarray): Solution value.

        Returns:
            float: Value of the level function.
        """
        alpha, beta = 1, 1
        sum_ = 0
        m2 = 0 if self.f2_func is None else self.f2_func(x).shape[0]
        if self.f2_func is not None:
            lambda_ = np.abs(self.adjoint_variables[:m2])
            f2 = np.abs(self.f2_func(x))
            alpha = np.maximum(lambda_, (1 / 2) * (alpha + lambda_))
            sum_ += np.sum(alpha * f2)
        if self.f3_func is not None:
            mu_ = np.abs(self.adjoint_variables[m2:])
            f3 = np.abs(np.minimum(0, self.f3_func(x)))
            beta = np.maximum(mu_, (1 / 2) * (beta + mu_))
            sum_ += np.sum(beta * f3)
        sum_ += (1 / 2) * np.linalg.norm(self.f1_func(x)) ** 2
        return sum_

    @staticmethod
    def where_zero(x: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        """
        Return indices of x where x is zero.

        Args:
            x (np.ndarray): Input array.
            tol (float, optional): Tolerance for zero. Defaults to 1e-6.

        Returns:
            np.ndarray: Indices of x where x is almost zero.
        """
        return np.where(np.isclose(x, 0, atol=tol))[0]

    @staticmethod
    def where_positive(x: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        """
        Return indices of x where x is positive.

        Args:
            x (np.ndarray): Input array.
            tol (float, optional): Tolerance for zero. Defaults to 1e-6.

        Returns:
            np.ndarray: Indices of x where x is positive.
        """
        return np.where(x > tol)[0]

    @staticmethod
    def where_negative(x: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        """
        Return indices of x where x is negative.

        Args:
            x (np.ndarray): Input array.
            tol (float, optional): Precision for negative. Defaults to 1e-6.

        Returns:
            np.ndarray: Indices of x where x is negative.
        """
        return np.where(x < -tol)[0]
