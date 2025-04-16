import autograd.numpy as np
from abc import abstractmethod, ABC
from dataclasses import dataclass


class BaseOptimizer(ABC):
    """
    Optimizer interface - not functional on its own.
    """

    @abstractmethod
    def __init__(self):
        """
        Initialize the optimizer.
        """
        self.result = OptimizerResult()
        self.result.level_functions = []

    @abstractmethod
    def minimize(self, method: str = None, options: dict = None):
        """
        Minimize the objective function subject to bounds and constraints.

        Args:
            method (str, optional): For consistency, non-functional for GaussNewton. Defaults to None.
            options (dict, optional): Solver options. Defaults to None.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError


@dataclass
class OptimizerResult:
    """
    Optimizer result object.

    Attributes:
        x (np.ndarray): Solution of the optimizer.
        success (bool): Whether the optimizer has converged.
        message (str): Cause of termination.
        func (np.ndarray): Objective (or level) function at solution.
        jac (np.ndarray): Value of the Jacobian at the solution.
        hess (np.ndarray): Value of the Hessian at the solution.
        hess_inv (np.ndarray): Value of the Hessian inverse at the solution.
        n_iters (int): Number of iterations performed.
        max_cv (float): Maximum constraint violation.
        level_functions (list): List of level functions.
    """

    x = None
    success = False
    message = "Optimization has not been attempted."
    func = None
    jac = None
    hess = None
    hess_inv = None
    n_iters = 0
    max_cv = np.inf
    level_functions = None
