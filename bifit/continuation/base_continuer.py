import scipy
import autograd.numpy as np
from typing import Tuple
from autograd import jacobian
from bisect import bisect, bisect_left


class Continuer:
    """
    Continuer interface, not functional on its own.

    The continuer takes a non-linear system, a known solution, and a parameter range and continues the solution.
    It returns the ContinuationSolution object.
    """

    def __init__(
        self,
        func: callable,
        x0: np.ndarray,
        p0: float = np.nan,
        p_min: float = 0,
        p_max: float = np.inf,
        p_step: float = 1,
        p_idx: int = -1,
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
            p_idx (int, optional): Index of the parameter in the input to `func`. Defaults to -1.
            data (np.ndarray, optional): Data points to trace. Defaults to None.
        """
        self.func = func
        self.x0 = x0
        self.p0 = p0
        self.p_min = p_min
        self.p_max = p_max
        self.p_step = p_step
        self.p_idx = p_idx
        self.data = data

        self._direction_str = {1: "forward", -1: "backward"}
        self._bisect_funcs = {1: bisect, -1: bisect_left}
        self.flag = False

    def _join_x_and_p(self, x: np.ndarray, p: float) -> np.ndarray:
        """
        Combine the solution vector `x` with the parameter `p`.

        Args:
            x (np.ndarray): Solution vector.
            p (float): Parameter value.

        Returns:
            np.ndarray: Combined solution.
        """
        axis = None if len(x.shape) == 1 else 1
        return np.insert(arr=x, obj=self.p_idx, values=p, axis=axis)

    def _check_if_solution_satisfies_ftol(self, x: np.ndarray, p: float, ftol: float):
        """
        Check if the solution satisfies the stopping criterion.

        Args:
            x (np.ndarray): Solution.
            p (float): Parameter value.
            ftol (float): Tolerance value.

        Returns:
            bool: True if the solution satisfies the stopping criterion, False otherwise.
        """
        solution = self._join_x_and_p(x, p)
        return np.linalg.norm(self.func(solution)) < ftol

    def _compute_jacobians(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Jacobians of the function with respect to x and p.

        Args:
            y (np.ndarray): Input to the function.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Jacobian of the function with respect to x and p.
        """
        J = jacobian(self.func)(y)
        Jx = np.delete(J, self.p_idx, axis=1)
        Jp = J[:, self.p_idx]
        return Jx, Jp

    @staticmethod
    def _solve_linear_system(A: np.ndarray, b: np.ndarray):
        """
        Solve a system of linear equations using a least squares solver.

        Args:
            A (np.ndarray): Coefficient matrix.
            b (np.ndarray): Right-hand side vector.

        Returns:
            np.ndarray: Solution to the system of linear equations.
        """
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            try:
                P, L, U = scipy.linalg.lu(A)
                Lhat, lhat = L[:-1, :-1], L[-1, :-1].T
                Uhat, uhat = U[:-1, :-1], U[:-1, -1]
                epsilon = U[-1, -1]
                rhs = -P.T @ b
                g, gamma = rhs[:-1], rhs[-1]
                Phi = np.linalg.solve(Lhat, g)
                phi = gamma - lhat.T @ Phi
                psi = phi / epsilon
                Psi = np.linalg.solve(Uhat, Phi - uhat * psi)
                x = np.hstack([Psi, psi])
            except Exception or RuntimeWarning:
                x = None
        return x

    def _trace_data(self, x: np.ndarray, dx: np.ndarray, step: float, direction: int) -> float:
        """
        Adjust step size to trace measurements.

        Args:
            x (np.ndarray): Initial value.
            dx (np.ndarray): Step vector.
            step (float): Step size.
            direction (int): Direction of continuation.

        Returns:
            float: Adjusted step size.
        """

        _bisect = self._bisect_funcs[direction]
        idx1 = _bisect(self.data, x)
        idx2 = _bisect(self.data, x + step * dx)
        if idx1 < idx2:
            x_tilde = self.data[idx1]
        elif idx1 > idx2:
            x_tilde = self.data[idx1 - 1]
        else:
            self.flag = False
            return step
        self.flag = True
        return (x_tilde - x) / dx
