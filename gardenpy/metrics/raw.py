r"""
**GardenPy raw data interpretation tools.**

Contains:
    - :func:`confusion_matrix`
"""

from typing import Union
import numpy as np


def c_matrix(pred: Union[list, np.ndarray], expect: Union[list, np.ndarray], *, norm: bool = True) -> np.ndarray:
    r"""
    **Confusion matrix.**

    Generates a confusion matrix from raw data.
    Uses labels that start from one, not zero.

    Args:
        pred (list, np.ndarray): Predicted outcomes.
        expect (list, np.ndarray): Expected outcomes.
        norm (bool): Toggle for normalization.

    Returns:
        np.ndarray: Generated confusion matrix.

    Raises:
        TypeError: Invalid predicted or expected raw data values.
        ValueError: Non-matching predicted and expected raw data values.
    """
    # reformat data
    pred, expect = np.array(pred).squeeze() - 1, np.array(expect).squeeze() - 1

    # check data
    if not np.issubdtype(pred.dtype, np.number) or len(pred.shape) != 1:
        raise TypeError("Attempted cm creation with predicted labels that were not one-dimensional numerical arrays.")
    if not np.issubdtype(expect.dtype, np.number) or len(expect.shape) != 1:
        raise TypeError("Attempted cm creation with expected labels that were not one-dimensional numerical arrays.")
    if len(pred) != len(expect):
        raise ValueError("Attempted cm creation with mismatching predicted and expected labels.")

    # generate cm
    pred, expect = list(pred), list(expect)
    cm = [[0 for _ in list(set(pred))] for _ in list(set(expect))]
    for pr, ex in zip(pred, expect):
        cm[ex][pr] += 1
    if not norm:
        return np.array(cm)

    # normalization
    for i, row in enumerate(cm):
        r_sum = np.sum(row)
        if r_sum != 0:
            cm[i] = list(np.array(row) / r_sum)
    return np.array(cm)
