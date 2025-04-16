import autograd.numpy as np
from autograd import jacobian
from scipy import sparse
import osqp
from ...optimization.multi_experiment.base_optimizer import BaseMultiExperimentOptimizer


class MultiExperimentOSQP(BaseMultiExperimentOptimizer):
    """
    Solves multi-experiment non-linear optimization problems using OSQP.

    Reference:
        Stellato, Bartolomeo, et al. "OSQP: An operator splitting solver for quadratic programs."
        Mathematical Programming Computation 12.4 (2020): 637-672.
    """

    def __init__(
        self,
        x0: np.ndarray,
        f1_fun: callable,
        f2_fun: callable,
        n_local: int,
        n_global: int,
        n_observables: int,
        n_experiments: int,
        xtol: float = 1e-3,
        ftol: float = 1e-3,
        max_iters: int = 100,
        plot_iters: bool = False,
        compute_ci: bool = False,
    ):
        """
        Initializes the OSQP optimizer.

        Args:
            x0 (np.ndarray): Initial guess.
            f1_fun (callable): Objective function.
            f2_fun (callable): Equality constraint.
            n_local (int): Number of local parameters.
            n_global (int): Number of global parameters.
            n_observables (int): Number of observables.
            n_experiments (int): Number of experiments.
            xtol (float): Convergence threshold for step size.
            ftol (float): Convergence threshold for function value.
            max_iters (int): Maximum number of iterations.
            plot_iters (bool): Whether to plot the level function at each iteration.
            compute_ci (bool): Whether to compute confidence intervals.

        Raises:
            AssertionError: If no global parameters are found.
        """

        super().__init__()
        self.x0 = x0
        self.f1_fun = f1_fun
        self.f2_fun = f2_fun
        self.n_local = n_local
        self.n_global = n_global
        self.n_observables = n_observables
        self.n_experiments = n_experiments
        self.xtol = xtol
        self.ftol = ftol
        self.max_iters = max_iters
        self.plot_iters = plot_iters
        self.compute_ci = compute_ci

        assert self.n_global > 0, (
            "No global parameters found. " "Multi-experiment PE does not make sense in this case!"
        )

        self.j1_fun = jacobian(self.f1_fun)
        self.j2_fun = jacobian(self.f2_fun)

        self.n_total_parameters = self.n_experiments * self.n_local + self.n_global

        self.j1 = np.array([]).reshape(0, self.n_total_parameters)
        self.j2 = np.array([]).reshape(0, self.n_total_parameters)

        y0 = self.split_into_experiments(x0)
        self.n_local_constr = len(self.f2_fun(y0[0]))

        self.alpha = [np.ones(self.n_local_constr * self.n_experiments, dtype=float)]

    def _solve_linearized_system(self) -> np.ndarray:
        """
        Solves the linearized system using OSQP.

        Returns:
            np.ndarray: Solution vector.
        """
        P = sparse.csc_matrix(self.j1.T @ self.j1)
        q = self.j1.T @ self.f1
        A = sparse.csc_matrix(self.j2)
        l, u = -self.f2, -self.f2

        problem = osqp.OSQP()
        problem.setup(P, q, A, l, u)
        res = problem.solve()

        self.lagrange_multipliers.append(res.y.copy())
        return res.x

    def _compute_covariance_matrix(self) -> np.ndarray:
        """
        Computes the covariance matrix at the solution.

        Returns:
            np.ndarray: Covariance matrix.
        """
        x = self.result.x
        self.jacobian_evaluation(x)

        n_con = self.j2.shape[0]
        n_var = self.j1.shape[1]

        I = np.eye(n_var)
        O1 = np.zeros((n_var, n_var))
        O2 = np.zeros((n_var, n_con))
        O3 = np.zeros((n_con, n_con))
        X = np.block([[self.j1.T @ self.j1, O2], [O2.T, O3]])
        Id = np.column_stack((I, O2))
        KKT = X + np.block([[O1, self.j2.T], [self.j2, O3]])
        KKT_inv = np.linalg.inv(KKT)

        return Id @ KKT_inv @ X @ KKT_inv.T @ Id.T
