import copy
import autograd.numpy as np
from ..continuation.base_continuer import Continuer
from ..optimization.line_search import line_search
from ..logging_ import logger


class PseudoArclengthContinuation(Continuer):
    """
    Implements the pseudo-arclength continuation method.
    """

    def __init__(
        self,
        func: callable,
        x0: np.ndarray,
        p0: float = np.nan,
        p_min: float = -np.inf,
        p_max: float = np.inf,
        p_step: float = 1,
        p_idx: int = -1,
        max_iters: int = 500,
        max_newton_iters: int = 10,
        newton_fun_tol: float = 1e-4,
        newton_var_tol: float = 1e-4,
        fast_iters: int = 3,
        data: np.ndarray = None,
    ):
        """
        Initialize the pseudo-arclength continuation method.

        Args:
            func (callable): Function of x and p.
            x0 (np.ndarray): Initial guess.
            p0 (float, optional): Initial value of the parameter. Defaults to np.nan.
            p_min (float, optional): Minimum value of the parameter. Defaults to -np.inf.
            p_max (float, optional): Maximum value of the parameter. Defaults to np.inf.
            p_step (float, optional): Step size of the parameter. Defaults to 1.
            p_idx (int, optional): Index of the parameter in the input to `func`. Defaults to -1.
            max_iters (int, optional): Maximum number of predictor-corrector iterations. Defaults to 500.
            max_newton_iters (int, optional): Maximum number of iterations for Newton corrector. Defaults to 10.
            newton_fun_tol (float, optional): Function tolerance for Newton corrector. Defaults to 1e-4.
            newton_var_tol (float, optional): Variable step tolerance for Newton corrector. Defaults to 1e-4.
            fast_iters (int, optional): Number of optimizer iterations for fast convergence. Defaults to 3.
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

        self.max_iters = max_iters
        self.max_newton_iters = max_newton_iters
        self.newton_fun_tol = newton_fun_tol
        self.newton_var_tol = newton_var_tol
        self.fast_iters = fast_iters

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

    def _compute_solutions(self, direction: int):
        """
        Find solutions in the given direction.

        Args:
            direction (int): Direction of continuation (1 for forward, -1 for backward).
        """
        y = self._join_x_and_p(x=self.x0, p=self.p0)

        p_min = self.p_min
        p_max = float(np.maximum(self.p0, self.p_max))
        step = self.p_step

        self._parameters = []
        self._solutions = []

        Jx, Jp = self._compute_jacobians(y)
        step_vector = self._solve_linear_system(A=Jx, b=-Jp)

        for i in range(self.max_iters):
            p = y[self.p_idx]
            x = np.delete(y, self.p_idx)

            dp = direction / np.sqrt(1 + (np.linalg.norm(step_vector) ** 2))
            dx = step_vector * dp
            success = False
            while not success and not np.isclose(dp, 0) and not np.isclose(step, 0):
                if self.data is not None:
                    step = self._trace_data(x=p, dx=dp, step=step, direction=direction)

                x_, p_, step, success = self._corrector_step(x0=x, dx0=dx, p0=p, dp0=dp, step=step)

                if success is True:
                    x, p = x_.copy(), p_

            if p < p_min or p > p_max or np.isclose(dp, 0) or np.isclose(step, 0):
                break

            logger.debug(
                f"Continued solution to parameter value: {p} in {self._direction_str[direction]} direction."
            )
            self._parameters.append(p)
            self._solutions.append(x)

            y = self._join_x_and_p(x=x, p=p)
            Jx, Jp = self._compute_jacobians(y)
            step_vector = self._solve_linear_system(A=Jx, b=-Jp)
            if np.sign(dx.T @ step_vector + dp) != direction:
                old_direction = self._direction_str[direction]
                new_direction = self._direction_str[np.sign(dx.T @ step_vector + dp)]
                logger.debug(f"Changing directions: {old_direction} -> {new_direction}")
            direction = np.sign(dx.T @ step_vector + dp)

    def _corrector_step(self, x0: np.ndarray, dx0: np.ndarray, p0: float, dp0: float, step: float):
        """
        Perform a corrector step to find a solution.

        Args:
            x0 (np.ndarray): Initial guess for x.
            dx0 (np.ndarray): Step vector for x.
            p0 (float): Initial guess for p.
            dp0 (float): Step for p.
            step (float): Step size for p.

        Returns:
            Tuple[np.ndarray, float, float, bool]: Solution for x, solution for p, updated step size, and success flag.
        """
        success = False
        x = x0 + step * dx0
        p = p0 + step * dp0

        if self.flag:
            func = lambda y_: 0.5 * np.linalg.norm(self.func(y_)) ** 2
        else:

            def func(y_):
                f1 = self.func(y_)
                x_ = np.concatenate((y_[: self.p_idx], y_[self.p_idx + 1 :]))
                p_ = y_[self.p_idx]
                f2 = (x_ - x0).T @ dx0 + (p_ - p0) * dp0 - step
                return 0.5 * np.linalg.norm(np.hstack((f1, f2))) ** 2

        for i in range(self.max_newton_iters):
            y = self._join_x_and_p(x=x, p=p)
            Jx, Jp = self._compute_jacobians(y)

            obj_func = self.func(y)
            if self.flag:
                dx = self._solve_linear_system(A=Jx, b=-obj_func)
                if dx is None:
                    break
                dp = 0
                dy = self._join_x_and_p(dx, dp)
            else:
                row1 = np.insert(Jx, self.p_idx, Jp, axis=1)
                row2 = np.insert(dx0, self.p_idx, dp0)
                coeff = np.row_stack((row1, row2))

                cont_func = (x - x0).T @ dx0 + (p - p0) * dp0 - step
                rhs = np.hstack((obj_func, cont_func))
                dy = self._solve_linear_system(A=coeff, b=-rhs)
                if dy is None:
                    break
                dx = np.delete(dy, self.p_idx)
                dp = dy[self.p_idx]

            fun_value = np.linalg.norm(self.func(self._join_x_and_p(x=x, p=p)))
            if fun_value < self.newton_fun_tol and np.linalg.norm(dy) < self.newton_var_tol:
                success = True
                if step <= 0.5 * self.p_step and i < self.fast_iters:
                    step = step * 2.0
                break

            t = line_search(x=y, dx=dy, func=func, strategy="armijo-backtracking")
            x = x + t * dx
            p = p + t * dp

        if not success:
            step = step / 2.0

        return x, p, step, success
