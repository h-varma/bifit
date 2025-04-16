import autograd.numpy as np
from abc import abstractmethod, ABC
from ..logging_ import logger


class BaseModel(ABC):
    """
    Base class for all models. This class is abstract and not functional on its own.

    Attributes:
        data (Any): Data associated with the model.
        data_weights (Any): Weights for the data.
        mask (dict): Mask indicating which components are active in the model.
    """

    def __init__(self):
        """
        Initializes the BaseModel with default attributes.
        """
        self.data = None
        self.data_weights = None
        self.mask = {
            "compartments": False,
            "controls": False,
            "auxiliary_variables": False,
            "global_parameters": False,
        }

    @abstractmethod
    def rhs_(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the right-hand side of the model equations.

        Args:
            x (np.ndarray): Model state (and parameters).

        Returns:
            np.ndarray: Right-hand side of the model equations.
        """
        raise NotImplementedError

    @abstractmethod
    def jacobian_(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Jacobian matrix of the model equations.

        Args:
            x (np.ndarray): Model state (and parameters).

        Returns:
            np.ndarray: Jacobian matrix of the model equations.
        """
        raise NotImplementedError

    def set_parameters(self, controls: dict = None, parameters: np.ndarray = None):
        """
        Generates random initial guesses for the variable parameters and controls.

        Args:
            controls (dict): Control parameters from the data.
            parameters (np.ndarray): Sample from the parameter space.
        """
        i = 0
        for _name, _value in self.true_parameters.items():
            if _name in self.global_parameters:
                if parameters is not None:
                    _value = float(parameters[i])
                i += 1
            elif _name in list(self.controls.values()):
                if controls is not None:
                    _value = controls[_name]
            self.initial_parameters[_name] = _value
            self.parameters[_name] = {"value": _value, "vary": False}

        logger.info(f"True model parameters: {self.true_parameters}")
        logger.info(f"Parameter guess initialization: {self.parameters}")

    def __getattr__(self, attr: str):
        """
        Retrieves attributes from the ProblemSpecifications object.

        Args:
            attr (str): Name of the attribute.

        Returns:
            Any: Value of the requested attribute.
        """
        return getattr(self.specifications, attr)
