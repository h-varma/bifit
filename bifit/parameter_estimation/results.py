def get_results(model: object, res: object = None):
    """
    Get a summary of the parameter estimation results.

    Args:
        model (object): Model object containing details of the system.
        res (object, optional): Parameter estimation results. Defaults to None.

    Returns:
        dict: A dictionary containing the results summary.
    """
    result = dict()

    result["data"] = {}
    result["data"]["values"] = model.data

    result["bifurcation"] = {}
    result["bifurcation"]["type"] = model.bifurcation_type
    result["bifurcation"]["continuation_settings"] = model.continuation_settings
    result["bifurcation"][
        "two_parameter_continuation_method"
    ] = model.two_parameter_continuation_method

    result["model"] = {}
    result["model"]["name"] = model.name
    result["model"]["true_parameters"] = model.true_parameters
    result["model"]["parameters"] = model.initial_parameters
    result["model"]["compartments"] = model.compartments
    result["model"]["controls"] = model.controls
    result["model"]["global_parameters"] = model.global_parameters
    result["model"]["multi_experiment_line_search"] = model.multi_experiment_line_search

    try:
        result["error_message"] = model.error_message
    except AttributeError:
        result["error_message"] = None

    if res is None:
        return result

    if res.__dict__ != {}:
        result["PE"] = {}
        result["PE"]["n_experiments"] = res.n_experiments
        result["PE"]["initial_guesses"] = res.x0
        result["PE"]["method"] = res.method
        result["PE"]["xtol"] = res.xtol
        result["PE"]["ftol"] = res.ftol
        result["PE"]["max_iters"] = res.max_iters
        result["PE"]["residual"] = res.model.measurement_error
        result["PE"]["initial_model_state"] = res.model.initial_state
        result["PE"]["integration_interval"] = res.model.integration_interval

        result["PE"]["result"] = {}
        result["PE"]["result"]["x"] = res.result.x
        result["PE"]["result"]["success"] = res.result.success
        result["PE"]["result"]["message"] = res.result.message
        result["PE"]["result"]["func"] = res.result.func
        result["PE"]["result"]["n_iters"] = res.result.n_iters
        result["PE"]["result"]["level_functions"] = res.result.level_functions
        if res.compute_ci and res.result.success:
            result["PE"]["result"]["covariance_matrix"] = res.result.covariance_matrix
            result["PE"]["result"]["confidence_intervals"] = res.result.confidence_intervals
        else:
            result["PE"]["result"]["covariance_matrix"] = None
            result["PE"]["result"]["confidence_intervals"] = None

    return result
