import os
import autograd.numpy as np
from bifit.parser.yaml_parser import YamlParser
from bifit.models.utils import nparray_to_dict
from bifit.models.base_model import BaseModel


class Model(BaseModel):
    """
    Wieczorek, Sebastian, et al.
    "Bifurcation transitions in an optically injected diode laser: theory and experiment."
    Optics communications 215.1-3 (2003): 125-134.

    Attributes
    ----------
    specifications : dataclass
        model information, parameters and settings

    Methods
    -------
    rhs_(x):
        defines the RHS of the model.

    jacobian_(x):
        defines the Jacobian matrix of the model.
    """

    def __init__(self):
        super().__init__()
        file_path = os.path.dirname(__file__)
        parser = YamlParser(file_path=file_path)
        self.specifications = parser.get_problem_specifications()

    def rhs_(self, x: np.ndarray) -> np.ndarray:
        """
        RHS of the model

        Parameters
        ----------
        x : np.ndarray
            model state (and parameters)

        Returns
        -------
        np.ndarray : RHS of the model
        """

        c, p, _ = nparray_to_dict(x=x, model=self)
        model_equations = {
            "Ex": p["K"]
            + 0.5 * c["n"] * c["Ex"]
            - 0.5 * p["alpha"] * c["n"] * c["Ey"]
            + p["omega"] * c["Ey"],
            "Ey": 0.5 * c["n"] * c["Ey"]
            + 0.5 * p["alpha"] * c["n"] * c["Ex"]
            - p["omega"] * c["Ex"],
            "n": -2 * p["gamma"] * c["n"]
            - (1 + 2 * p["B"] * c["n"]) * ((c["Ex"] ** 2) + (c["Ey"] ** 2) - 1),
        }

        M_list = [model_equations[key] for key in self.compartments]
        return np.array(M_list)

    def jacobian_(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian matrix of the model

        Parameters
        ----------
        x : np.ndarray
            model state (and parameters)

        Returns
        -------
        np.ndarray : Jacobian matrix of the model
        """

        c, p, _ = nparray_to_dict(x=x, model=self)
        model_jacobian = {
            "Ex": np.array(
                [
                    0.5 * c["n"],
                    -0.5 * p["alpha"] * c["n"] + p["omega"],
                    0.5 * c["Ex"] - 0.5 * p["alpha"] * c["Ey"],
                ]
            ),
            "Ey": np.array(
                [
                    0.5 * p["alpha"] * c["n"] - p["omega"],
                    0.5 * c["n"],
                    0.5 * c["Ey"] + 0.5 * p["alpha"] * c["Ex"],
                ]
            ),
            "n": np.array(
                [
                    -(1 + 2 * p["B"] * c["n"]) * 2 * c["Ex"],
                    -(1 + 2 * p["B"] * c["n"]) * 2 * c["Ey"],
                    -2 * p["gamma"] - 2 * p["B"] * (c["Ex"] ** 2 + c["Ey"] ** 2 - 1),
                ]
            ),
        }

        J_list = [model_jacobian[key] for key in self.compartments]
        return np.row_stack(J_list)
