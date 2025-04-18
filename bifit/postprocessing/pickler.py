import os
import pickle
import easygui
from datetime import datetime
from bifit.estimation.results import get_results


def create_folder_for_results(path: str):
    """
    Creates a folder for storing the results.

    Args:
        path (str): Path to the model folder.

    Returns:
        str: Path to the newly created folder.

    Raises:
        FileExistsError: If the folder already exists.
    """
    folder_path = os.path.join(path, "results")
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S%f")
    timestamped_path = os.path.join(folder_path, timestamp)
    try:
        os.mkdir(timestamped_path)
    except FileExistsError:
        raise FileExistsError(f"The folder {timestamp} already exists.")

    return timestamped_path


def save_results_as_pickle(model: object, res: object, path: str):
    """
    Pickles the parameter estimation results.

    Args:
        model (object): The model object.
        res (object): The parameter estimation results.
        path (str): Path to the folder where the results are to be stored.
    """
    with open(os.path.join(path, "summary.pkl"), "wb") as f:
        pickle.dump(get_results(model=model, res=res), f)


def load_results_from_pickle():
    """
    Loads the parameter estimation results from a pickle file.

    Returns:
        dict: The results.
    """
    file_path = easygui.fileopenbox()
    with open(file_path, "rb") as f:
        return pickle.load(f)
