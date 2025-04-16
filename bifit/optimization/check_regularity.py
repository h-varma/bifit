import autograd.numpy as np


def check_positive_definiteness(J: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Checks if the matrix has full column rank.

    Args:
        J (np.ndarray): Matrix to check.
        tol (float, optional): Singular value tolerance. Defaults to 1e-6.

    Returns:
        bool: True if the matrix has full column rank, False otherwise.
    """
    return np.linalg.matrix_rank(J, tol=tol) == J.shape[1]


def check_constraint_qualification(J: np.ndarray) -> bool:
    """
    Checks if the matrix has full row rank.

    Args:
        J (np.ndarray): Matrix to check.

    Returns:
        bool: True if the matrix has full row rank.

    Raises:
        Exception: If the constraint qualification (CQ) fails.
    """
    if len(J):
        if np.linalg.matrix_rank(J) != J.shape[0] and len(J.shape) > 1:
            raise Exception("CQ failed!")
    return True
