import yaml
import autograd.numpy as np
from dataclasses import dataclass
from ..models.specifications import ProblemSpecifications
from ..logging_ import logger


class YamlParser:
    """
    A class to parse YAML files and extract problem specifications.

    Attributes:
        meta_params_path (str): Path to the meta parameters YAML file.
        meta_parameters (dict): Parsed meta parameters.
        specifications (ProblemSpecifications): Parsed problem specifications.
    """

    def __init__(self, file_path):
        """
        Initializes the YamlParser with the file paths.

        Args:
            file_path (str): The base path to the YAML files.
        """
        self.meta_params_path = file_path + "/meta_parameters.yaml"
        self.meta_parameters = None
        self.specifications = None

    def __load_yaml(self, file_path):
        """
        Loads a YAML file.

        Args:
            file_path (str): Path to the YAML file.

        Returns:
            dict: Parsed YAML data.

        Raises:
            FileNotFoundError: If the file is not found.
            ValueError: If there is an error parsing the YAML file.
        """
        try:
            logger.info(f"Loading YAML file: {file_path}")
            with open(file_path, "r") as file:
                data = yaml.safe_load(file)
                logger.info(f"Successfully loaded YAML file.")
                return data
        except FileNotFoundError:
            logger.error(f"The file {file_path} was not found.")
            raise FileNotFoundError(f"The file {file_path} was not found.")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {file_path}: {e}")
            raise ValueError(f"Error parsing YAML file {file_path}: {e}")

    def __load_meta_parameters(self):
        """
        Loads the meta parameters from the YAML file.

        Returns:
            dict: The meta parameters.
        """
        logger.info("Parsing meta parameters.")
        self.meta_parameters = self.__load_yaml(self.meta_params_path)

    def __parse_problem_specifications(self) -> ProblemSpecifications:
        """
        Parses the problem specifications from the YAML files.

        Returns:
            ProblemSpecifications: The parsed problem specifications.

        Raises:
            AssertionError: If the model names in the specifications and meta parameters do not match.
        """
        kwargs = {}
        kwargs["name"] = self.meta_parameters["model_name"]

        kwargs["compartments"] = []
        kwargs["initial_state"] = []
        kwargs["true_parameters"] = {}
        kwargs["initial_parameters"] = {}
        kwargs["parameters"] = {}
        kwargs["global_parameters"] = []
        control_1 = {}
        control_2 = {}

        for key, values in self.meta_parameters.items():
            if key.startswith("compartment"):
                kwargs["compartments"].append(values["name"])
                kwargs["initial_state"].append(values["value"])

            elif key.startswith("parameter"):
                p_name = values["name"]
                true_p = values["default_value"]
                kwargs["true_parameters"][p_name] = true_p

                try:
                    initial_p = values["initial_value"]
                except KeyError:
                    initial_p = true_p
                kwargs["initial_parameters"][p_name] = initial_p
                kwargs["parameters"][p_name] = {"value": initial_p, "vary": False}

                if values["type"] == "control_1":
                    control_1["name"] = values["name"]
                    control_1["min_value"] = values["min_value"]
                    control_1["max_value"] = values["max_value"]
                    control_1["step_size"] = values["step_size"]

                if values["type"] == "control_2":
                    control_2["name"] = values["name"]

                if values["type"] == "global":
                    kwargs["global_parameters"].append(values["name"])

        kwargs["to_plot"] = self.meta_parameters["to_plot"]

        kwargs["integration_interval"] = np.array([0, self.meta_parameters["t_end"]])
        kwargs["bifurcation_type"] = self.meta_parameters["bifurcation"]["type"]

        kwargs["controls"] = {"homotopy": control_1["name"], "free": control_2["name"]}

        kwargs["continuation_settings"] = {
            "h_min": control_1["min_value"],
            "h_max": control_1["max_value"],
            "h_step": control_1["step_size"],
        }

        kwargs["two_parameter_continuation_method"] = self.meta_parameters[
            "two_parameter_continuation_method"
        ]

        residual = self.meta_parameters["residual"]
        kwargs["measurement_error"] = residual["type"] + "_" + residual["scale"]

        kwargs["exclude_states"] = []
        try:
            for state in self.meta_parameters["exclude"].values():
                kwargs["exclude_states"].append(np.array(state))
        except KeyError:
            pass

        try:
            line_search = self.meta_parameters["multi_experiment_line_search"]
        except KeyError:
            line_search = "armijo-backtracking"

        kwargs["multi_experiment_line_search"] = line_search
        return ProblemSpecifications(**kwargs)

    def get_problem_specifications(self):
        """
        Retrieves all the problem specifications from the relevant YAML files.

        Returns:
            ProblemSpecifications: The problem specifications.
        """
        if self.meta_parameters is None:
            logger.info("Meta parameters not loaded yet. Parsing now.")
            self.__load_meta_parameters()

        self.specifications = self.__parse_problem_specifications()
        return self.specifications
