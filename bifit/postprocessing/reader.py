import os
import sys
import pickle
import easygui
import numpy as np
import matplotlib.pyplot as plt


def read_results():
    """
    Reads the pickled results from the directory selected by the user.

    Returns:
        list: A list of results loaded from pickle files.
    """
    current_path = os.path.dirname(__file__)
    sys.path.append(os.path.abspath(os.path.join(current_path, "../..")))
    folder_path = easygui.diropenbox()

    results = []
    for name in os.listdir(folder_path):
        try:
            with open(os.path.join(folder_path, name, "summary.pkl"), "rb") as f:
                results.append(pickle.load(f))
        except FileNotFoundError:
            pass
    return results


def plot_convergence(results: list[dict], error: float = 0.05):
    """
    Plots the convergence area of the parameter estimation runs.

    Args:
        results (list[dict]): List of dictionaries containing the results of the parameter estimation.
        error (float): The error threshold for the convergence check (percentage of the true value).
    """

    p_global = results[0]["model"]["global_parameters"]
    p_true = results[0]["model"]["true_parameters"]

    n_global = len(p_global)

    fig, ax = plt.subplots()

    # Plot the true parameters
    true_parameters = [p_true[key] for key in p_global]
    ax.scatter(*true_parameters, marker="*", color="black", s=plt.rcParams["lines.markersize"] ** 3)

    for k, res in enumerate(results):
        model = res["model"]
        model_parameters = res["model"]["parameters"]
        # Check if the parameters are consistent for all the runs in the list
        assert p_true == model["true_parameters"], "True parameters are not the same!"
        assert p_global == model["global_parameters"], "Global parameters are not the same!"

        initial_parameters = [model_parameters[key] for key in p_global]

        # Default marker style for failed runs
        style = {
            "color": np.array([0.8, 0.4, 0, 1]).reshape(1, -1),
            "marker": "x",
        }
        try:
            if res["PE"]["result"]["success"]:
                initial_guess = res["PE"]["initial_guesses"]
                full_solution = res["PE"]["result"]["x"]

                # Check if the solution is within the error threshold
                success = True
                for i, solution in enumerate(full_solution[-n_global:]):
                    if np.abs(solution - true_parameters[i]) > error * true_parameters[i]:
                        success = False

                if success:
                    print(k, initial_guess[-n_global:], full_solution[-n_global:])
                    color = np.array([0, 0.6, 0.5, 1]).reshape(1, -1)
                else:
                    color = np.array([0, 0.6, 0.5, 0.5]).reshape(1, -1)

                style = {"color": color, "marker": "o"}
        except KeyError:
            pass

        ax.scatter(*initial_parameters, marker=style["marker"], color=style["color"])
    ax.set_xlabel(p_global[0], fontsize=14)
    ax.set_ylabel(p_global[1], fontsize=14)
    plt.show()
