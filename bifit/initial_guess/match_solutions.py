import autograd.numpy as np
from scipy.spatial.distance import cdist
from ..models.utils import nparray_to_dict
from itertools import chain


def match_solutions_to_data(model: object, solutions: list, fill_missing: bool) -> np.ndarray:
    """
    Matches the solutions of two-parameter continuation to the experimental data.

    Args:
        model (object): Details of the model.
        solutions (list): Solutions from the two-parameter continuation.
        fill_missing (bool): Whether to fill the missing data points.

    Returns:
        np.ndarray: Initial guess.

    Raises:
        RuntimeError: If no solutions are found in the two-parameter continuation.
        ValueError: If more than 2 data points per homotopy parameter value are encountered.
        AssertionError: If matching solutions are not found for all data or if there is a mismatch
            between homotopy values in initial guesses and data.
    """
    if len(solutions) == 0:
        raise RuntimeError("No solutions found in the two-parameter continuation!")

    h_param = model.controls["homotopy"]
    f_param = model.controls["free"]

    model.mask = {
        "compartments": True,
        "controls": True,
        "auxiliary_variables": True,
        "global_parameters": False,
    }
    model.parameters[h_param]["vary"] = True
    model.parameters[f_param]["vary"] = True

    if isinstance(solutions[0], list):
        solutions = list(chain.from_iterable(solutions))
    solutions = sorted(solutions, key=lambda x: get_parameter_value(x, type_="f", model=model))
    f_params = list(map(lambda x: get_parameter_value(x, type_="f", model=model), solutions))
    h_params = list(map(lambda x: get_parameter_value(x, type_="h", model=model), solutions))

    h_data = np.array([d[h_param] for d in model.data])
    f_data = np.array([d[f_param] for d in model.data])

    h_data_weights = np.array([d[h_param] for d in model.data_weights])
    f_data_weights = np.array([d[f_param] for d in model.data_weights])

    initial_guess = []
    data = []
    data_weights = []

    unique_h_data, unique_idx = np.unique(h_data, return_index=True)
    for h_value, h_weight in zip(h_data[unique_idx], h_data_weights[unique_idx]):
        where_h = np.where(np.isclose(h_params, h_value))[0]

        _f_data = f_data[np.isclose(h_data, h_value)]
        _f_data_weights = f_data_weights[np.isclose(h_data, h_value)]

        if fill_missing:
            number_of_points = len(_f_data)
            if number_of_points > 2:
                raise ValueError(
                    "Cannot handle more than 2 data points per homotopy parameter value!"
                )

            match number_of_points:
                case 1:
                    match len(where_h):
                        case 0:
                            where_h = np.nanargmin(np.abs(h_params - h_value), keepdims=True)
                        case 1:
                            pass
                        case 2:
                            pass
                case 2:
                    match len(where_h):
                        case 0:
                            i_vals = np.row_stack((h_params, f_params))
                            for k in range(2):
                                d_vals = np.array([h_value, _f_data[k]]).reshape(-1, 1)
                                distance = np.linalg.norm(i_vals - d_vals, axis=0)
                                new_index = np.nanargmin(distance, keepdims=True)
                                where_h = np.append(where_h, new_index)
                        case 1:
                            _f_values = np.array(f_params)[where_h]
                            dist_matrix = cdist(_f_data.reshape(-1, 1), _f_values.reshape(-1, 1))
                            row, col = np.argwhere(dist_matrix == np.nanmin(dist_matrix))[0]
                            k = 0 if row == 1 else 1
                            i_vals = np.row_stack((h_params, f_params))
                            d_vals = np.array([h_value, _f_data[k]]).reshape(-1, 1)
                            distance = np.linalg.norm(i_vals - d_vals, axis=0)
                            new_index = np.nanargmin(distance, keepdims=True)
                            where_h = np.append(where_h, new_index)
                        case 2:
                            pass

        _f_values = np.array(f_params)[where_h]
        dist_matrix = cdist(_f_data.reshape(-1, 1), _f_values.reshape(-1, 1))

        for _ in range(min(dist_matrix.shape)):
            row, col = np.argwhere(dist_matrix == np.nanmin(dist_matrix))[0]
            dist_matrix[:, col] = np.nan
            dist_matrix[row, :] = np.nan

            for i, (hp, fp) in enumerate(zip(h_params, f_params)):
                if i in where_h and np.isclose(fp, _f_values[col]):
                    idx = i
            initial_guess.append(solutions[idx])

            data.append({h_param: h_value, f_param: _f_data[row]})
            data_weights.append({h_param: h_weight, f_param: _f_data_weights[row]})

    if fill_missing:
        assert len(initial_guess) == len(h_data), "Matching solutions not found for all data!"
    else:
        assert len(initial_guess) > 0, "No matching solutions found for the experimental data!"
        hp = list(map(lambda x: get_parameter_value(x, type_="h", model=model), initial_guess))
        hd = [d[h_param] for d in data]
        if not all([np.isclose(hp, hd) for hp, hd in zip(hp, hd)]):
            raise AssertionError("Mismatch between homotopy values in initial guesses and data!")

    if not len(initial_guess) == len(data):
        raise AssertionError("Length of initial guesses and data do not match!")

    model.data = data
    model.data_weights = data_weights
    return np.hstack(initial_guess)


def get_parameter_value(x: np.ndarray, type_: str, model: object) -> float:
    """
    Gets the value of homotopy or free parameter from the solution array.

    Args:
        x (np.ndarray): Solution array.
        type_ (str): Parameter type ("h" for homotopy, "f" for free).
        model (object): Details of the model.

    Returns:
        float: Parameter value.

    Raises:
        ValueError: If the parameter type is unrecognized.
    """
    if type_ == "h":
        type_ = "homotopy"
    elif type_ == "f":
        type_ = "free"
    else:
        raise ValueError("Unrecognized parameter type!")

    _, p, _ = nparray_to_dict(x, model=model)
    return p[model.controls[type_]]
