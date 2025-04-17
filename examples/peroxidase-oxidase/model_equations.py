import os
import autograd.numpy as np
from bifit.parser.yaml_parser import YamlParser
from bifit.models.utils import nparray_to_dict
from bifit.models.base_model import BaseModel


class Model(BaseModel):
    """
    Steinmetz, Curtis G., Raima Larter, and Baltazar D. Aguda.
    "Modelling a Biochemical Oscillator."
    Proceedings of the Indiana Academy of Science. Vol. 98. 1988.

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
            "A": -p["k1"] * c["A"] * c["B"] * c["X"]
            - p["k3"] * c["A"] * c["B"] * c["Y"]
            + p["k7"]
            - p["km7"] * c["A"],
            "B": -p["k1"] * c["A"] * c["B"] * c["X"] - p["k3"] * c["A"] * c["B"] * c["Y"] + p["k8"],
            "X": p["k1"] * c["A"] * c["B"] * c["X"]
            - 2 * p["k2"] * (c["X"] ** 2)
            + 2 * p["k3"] * c["A"] * c["B"] * c["Y"]
            - p["k4"] * c["X"]
            + p["k6"],
            "Y": -p["k3"] * c["A"] * c["B"] * c["Y"]
            + 2 * p["k2"] * (c["X"] ** 2)
            - p["k5"] * c["Y"],
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
            "A": np.array(
                [
                    -p["k1"] * c["B"] * c["X"] - p["k3"] * c["B"] * c["Y"] - p["km7"],
                    -p["k1"] * c["A"] * c["X"] - p["k3"] * c["A"] * c["Y"],
                    -p["k1"] * c["A"] * c["B"],
                    -p["k3"] * c["A"] * c["B"],
                ]
            ),
            "B": np.array(
                [
                    -p["k1"] * c["B"] * c["X"] - p["k3"] * c["B"] * c["Y"],
                    -p["k1"] * c["A"] * c["X"] - p["k3"] * c["A"] * c["Y"],
                    -p["k1"] * c["A"] * c["B"],
                    -p["k3"] * c["A"] * c["B"],
                ]
            ),
            "X": np.array(
                [
                    p["k1"] * c["B"] * c["X"] + 2 * p["k3"] * c["B"] * c["Y"],
                    p["k1"] * c["A"] * c["X"] + 2 * p["k3"] * c["A"] * c["Y"],
                    p["k1"] * c["A"] * c["B"] - 4 * p["k2"] * c["X"] - p["k4"],
                    2 * p["k3"] * c["A"] * c["B"],
                ]
            ),
            "Y": np.array(
                [
                    -p["k3"] * c["B"] * c["Y"],
                    -p["k3"] * c["A"] * c["Y"],
                    4 * p["k2"] * c["X"],
                    -p["k3"] * c["A"] * c["B"] - p["k5"],
                ]
            ),
        }

        J_list = [model_jacobian[key] for key in self.compartments]
        return np.row_stack(J_list)
