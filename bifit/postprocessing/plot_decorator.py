import matplotlib.pyplot as plt
from functools import wraps

# Global flags, modifiable from the main script
save_plots = False
show_plots = True
save_path = None


def handle_plots(plot_name: str):
    """
    Decorator to handle the saving and showing of plots.

    Args:
        plot_name (str): Name of the plot.

    Returns:
        callable: The decorated function.
    """

    def decorator(func):
        @wraps(func)
        def wrapper_plot(*args, **kwargs):
            global save_plots, show_plots, save_path
            sol, fig = func(*args, **kwargs)

            if save_plots:
                file_path = save_path + f"/{plot_name}.png"
                fig.savefig(file_path)
                print(f"Plot saved to {save_path}")

            if show_plots:
                plt.show(block=True)

            plt.close(fig)
            return sol, fig

        return wrapper_plot

    return decorator
