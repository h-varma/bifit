import autograd.numpy as np
from scipy.stats import qmc
from scipy.stats import truncnorm


def generate_samples_using_lhs(parameters: dict, bounds: dict, n_points: int = 100):
    """
    Generates parameter samples using Latin Hypercube Sampling.

    Args:
        parameters (dict): Dictionary of parameters.
        bounds (dict): Dictionary of bounds for the parameters.
        n_points (int): Number of points to generate.

    Returns:
        np.ndarray: Scaled samples.
    """
    l_bounds, u_bounds = [], []
    for parameter_name in parameters.keys():
        if parameter_name in bounds.keys():
            l_bounds.append(bounds[parameter_name][0])
            u_bounds.append(bounds[parameter_name][1])

    sampler = qmc.LatinHypercube(d=len(l_bounds))
    samples = sampler.random(n=n_points)
    return qmc.scale(samples, l_bounds, u_bounds)


def generate_samples_using_gaussian(
    parameters: dict, to_vary: list, noise: float, n_points: int = 100, truncated: bool = False
):
    """
    Generates parameter samples using a Gaussian distribution.

    Args:
        parameters (dict): Dictionary of parameters.
        to_vary (list): List of parameters to vary.
        noise (float): Noise level.
        n_points (int): Number of points to generate.
        truncated (bool): Whether to generate truncated normal random variables.

    Returns:
        np.ndarray: Samples.
    """
    samples = np.zeros((n_points, len(to_vary)))
    i = 0
    for _name, _value in parameters.items():
        if _name in to_vary:
            scale = np.abs(_value * noise)
            if truncated:
                a = (0 - _value) / scale
                b = (np.inf - _value) / scale
                # compute truncated normal random variable
                # truncated at a and b standard deviations from loc
                samples[:, i] = truncnorm.rvs(a=a, b=b, loc=_value, scale=scale, size=n_points)
            else:
                samples[:, i] = np.random.normal(loc=_value, scale=scale, size=n_points)
            i += 1
    return samples
