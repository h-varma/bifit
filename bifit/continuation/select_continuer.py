def import_continuer(method: str):
    """
    Import a continuation method of choice.

    Args:
        method (str): Name of the continuation method.

    Returns:
        object: Continuation object.

    Raises:
        ValueError: If the continuation method is unknown.
    """
    if method == "deflated":
        from ..continuation.deflated_continuation import DeflatedContinuation

        return DeflatedContinuation

    elif method == "pseudo-arclength":
        from ..continuation.pseudo_arclength import PseudoArclengthContinuation

        return PseudoArclengthContinuation

    else:
        raise ValueError(f"Unknown continuation method: {method}")
