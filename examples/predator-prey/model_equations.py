import os
import autograd.numpy as np
from bifit.parser.yaml_parser import YamlParser
from bifit.models.utils import nparray_to_dict
from bifit.models.base_model import BaseModel


class Model(BaseModel):
    """
    Fussmann, Gregor F., et al.
    "Crossing the Hopf bifurcation in a live predator-prey system."
    Science 290.5495 (2000): 1358-1360.

    Attributes
    ----------
    specifications : dataclass
        model information, parameters and settings

    Methods
    -------
    model_rhs(x):
        defines the RHS of the model.

    model_jacobian(x):
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
            "N": p["delta"] * (p["Ni"] - c["N"]) - p["Bc"] * c["N"] * c["C"] / (p["Kc"] + c["N"]),
            "C": p["Bc"] * c["N"] * c["C"] / (p["Kc"] + c["N"])
            - p["Bb"] * c["C"] * c["B"] / (p["epsilon"] * (p["Kb"] + c["C"]))
            - p["delta"] * c["C"],
            "R": p["Bb"] * c["C"] * c["R"] / (p["Kb"] + c["C"])
            - (p["delta"] + p["m"] + p["lambda"]) * c["R"],
            "B": p["Bb"] * c["C"] * c["R"] / (p["Kb"] + c["C"]) - (p["delta"] + p["m"]) * c["B"],
        }

        M_list = [model_equations[key] for key in self.compartments]
        return np.array(M_list)

    def jacobian_(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian matrix of the model equations

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
            "N": np.array(
                [
                    -p["delta"] - (p["Bc"] * p["Kc"] * c["C"]) / ((p["Kc"] + c["N"]) ** 2),
                    -p["Bc"] * c["N"] / (p["Kc"] + c["N"]),
                    0,
                    0,
                ]
            ),
            "C": np.array(
                [
                    (p["Bc"] * p["Kc"] * c["C"]) / ((p["Kc"] + c["N"]) ** 2),
                    p["Bc"] * c["N"] / (p["Kc"] + c["N"])
                    - (p["Bb"] * p["Kb"] * c["B"]) / (p["epsilon"] * ((p["Kb"] + c["C"]) ** 2))
                    - p["delta"],
                    0,
                    -p["Bb"] * c["C"] / (p["epsilon"] * (p["Kb"] + c["C"])),
                ]
            ),
            "R": np.array(
                [
                    0,
                    p["Bb"] * p["Kb"] * c["R"] / ((p["Kb"] + c["C"]) ** 2),
                    p["Bb"] * c["C"] / (p["Kb"] + c["C"]) - (p["delta"] + p["m"] + p["lambda"]),
                    0,
                ]
            ),
            "B": np.array(
                [
                    0,
                    p["Bb"] * p["Kb"] * c["R"] / ((p["Kb"] + c["C"]) ** 2),
                    p["Bb"] * c["C"] / (p["Kb"] + c["C"]),
                    -(p["delta"] + p["m"]),
                ]
            ),
        }

        J_list = [model_jacobian[key] for key in self.compartments]
        return np.row_stack(J_list)
