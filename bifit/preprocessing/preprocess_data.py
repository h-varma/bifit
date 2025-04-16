import autograd.numpy as np
import pandas as pd
from itertools import chain
from ..logging_ import logger

rng = np.random.default_rng(0)


class DataPreprocessor:
    """
    A class to preprocess measurement data.

    Attributes:
        _data (pd.DataFrame or list): The measurement data.
        _weights (list): The weights associated with the data.
    """

    def __init__(self):
        """
        Initializes the DataPreprocessor with default values.
        """
        self._data = None
        self._weights = None

    def load_the_data(self, file_path: str, error_scale: float = 0.05):
        """
        Loads the measurement data from a file.

        Args:
            file_path (str): The path to the data file.
            error_scale (float): The scale of the measurement error.
        """
        if self._data is not None:
            logger.warning("Data already loaded. Overwriting the data.")

        self._data = pd.read_table(file_path + "/data.dat", sep=" ")

        control_1, control_2 = self._data.columns
        data = [[], []]
        errors = [[], []]
        for d1, d2 in zip(self._data[control_1], self._data[control_2]):
            d1_scale = np.abs(error_scale * d1)
            if type(d2) is str:
                for i, c2_ in enumerate(d2.split(",")):
                    d2_scale = np.abs(error_scale * float(c2_))
                    data[i].append({control_1: d1, control_2: float(c2_)})
                    errors[i].append({control_1: d1_scale, control_2: d2_scale})
            else:
                d2_scale = np.abs(error_scale * d2)
                data[0].append({control_1: d1, control_2: d2})
                errors[0].append({control_1: d1_scale, control_2: d2_scale})

        self._data = list(chain(*data))
        self._weights = list(chain(*errors))
        logger.info(f"Loaded {len(self._data)} data points from {file_path}.")

    def load_the_data_and_add_noise(self, file_path: str, error_scale: float = 0.05):
        """
        Loads the measurement data from a file and adds noise to the data.

        Args:
            file_path (str): The path to the data file.
            error_scale (float): The scale of the measurement error.
        """
        if self._data is not None:
            logger.warning("Data already loaded. Overwriting the data.")

        self._data = pd.read_table(file_path + "/data.dat", sep=" ")

        control_1, control_2 = self._data.columns
        noisy_data = [[], []]
        errors = [[], []]
        for d1, d2 in zip(self._data[control_1], self._data[control_2]):
            d1_scale = np.abs(error_scale * d1)
            d1 = rng.normal(d1, d1_scale)
            if type(d2) is str:
                for i, c2_ in enumerate(d2.split(",")):
                    d2_scale = np.abs(error_scale * float(c2_))
                    c2_ = rng.normal(float(c2_), d2_scale)
                    noisy_data[i].append({control_1: d1, control_2: c2_})
                    errors[i].append({control_1: d1_scale, control_2: d2_scale})
            else:
                d2_scale = np.abs(error_scale * d2)
                d2 = rng.normal(d2, d2_scale)
                noisy_data[0].append({control_1: d1, control_2: d2})
                errors[0].append({control_1: d1_scale, control_2: d2_scale})

        self._data = list(chain(*noisy_data))
        self._weights = list(chain(*errors))
        logger.info(f"Loaded {len(self._data)} data points with {error_scale * 100}% noise.")

    def select_subset_of_data(self, length: int):
        """
        Filters out a subset of the measurement data.

        Args:
            length (int): The number of data points to be selected.

        Raises:
            Exception: If the data is not a list or pandas DataFrame.
        """
        if length > len(self._data):
            logger.warning(
                f"{length} data points not available. " f"Using all available data points."
            )

        elif length is not None:
            filter_idx = np.random.choice(len(self._data), size=length, replace=False)
            filter_idx = np.sort(filter_idx)

            if type(self._data) is pd.DataFrame:
                self._data = self._data.iloc[filter_idx]
            elif type(self._data) is list:
                self._data = [d for i, d in enumerate(self._data) if i in filter_idx]
            else:
                raise Exception("Data must be a list or pandas DataFrame.")

        logger.info(f"Selected {len(self._data)} data points.")

    @property
    def data(self):
        """
        Gets the processed data.

        Returns:
            pd.DataFrame or list: The processed data.
        """
        return self._data

    @property
    def weights(self):
        """
        Gets the weights associated with the data.

        Returns:
            list: The weights associated with the data.
        """
        return self._weights
