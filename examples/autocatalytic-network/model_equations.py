import os
import autograd.numpy as np
from bifit.parser.yaml_parser import YamlParser
from bifit.models.utils import nparray_to_dict
from bifit.models.base_model import BaseModel


class Model(BaseModel):
    """
    Semenov, Sergey N., et al.
    "Autocatalytic, bistable, oscillatory networks of biologically relevant organic reactions."
    Nature 537.7622 (2016): 656-660.

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
            "A": p["k1"] * c["S"] * c["A"]
            - p["k2"] * c["I"] * c["A"]
            - p["k3"] * c["A"]
            - p["k0"] * c["A"]
            + p["k4"] * c["S"],
            "I": p["k0"] * p["I0"] - p["k0"] * c["I"] - p["k2"] * c["I"] * c["A"],
            "S": p["k0"] * p["S0"]
            - p["k0"] * c["S"]
            - p["k4"] * c["S"]
            - p["k1"] * c["S"] * c["A"],
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
                    p["k1"] * c["S"] - p["k2"] * c["I"] - p["k3"] - p["k0"],
                    -p["k2"] * c["A"],
                    p["k1"] * c["A"] + p["k4"],
                ]
            ),
            "I": np.array(
                [
                    -p["k2"] * c["I"],
                    -p["k0"] - p["k2"] * c["A"],
                    0,
                ]
            ),
            "S": np.array(
                [
                    -p["k1"] * c["S"],
                    0,
                    -p["k0"] - p["k4"] - p["k1"] * c["A"],
                ]
            ),
        }

        J_list = [model_jacobian[key] for key in self.compartments]
        return np.row_stack(J_list)
