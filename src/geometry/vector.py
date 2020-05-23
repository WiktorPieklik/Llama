import math

import numpy as np


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def calc_angle_inner(v1: np.ndarray, v2: np.ndarray):
    """ Returns angle between vector 1 and vector 2, expressed in degrees.

    Parameters
    ----------
    v1 : np.ndarray
    v2 : np.ndarray

    Returns
    -------
    float
        Angle between vector 1 and vector 2, expressed in degrees.
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return math.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def calc_two_vector_determinant(v1: np.ndarray, v2: np.ndarray):
    return v1[0] * v2[1] - v1[1] * v2[0]


def calc_angle_clockwise(v1: np.ndarray, v2: np.ndarray):
    inner_angle_deg = calc_angle_inner(v1, v2)
    det = calc_two_vector_determinant(v1, v2)
    if det < 0:
        return inner_angle_deg
    return 360 - inner_angle_deg


def convert_bound_to_free(vector_bound: np.ndarray):
    return vector_bound[1] - vector_bound[0]
