import dlib
import numpy as np


def dlib_point_to_np_array(point: dlib.point):
    """ Converts dlib.point instance to numpy array, as [x,y].

    Parameters
    ----------
    point : dlib.point
        The point to convert

    Returns
    -------
    np.ndarray
        Specified point converted to numpy array.
    """
    return np.array([point.x, point.y])


def md_point_to_np_array(point: dict):
    """ Converts metadata point to numpy array, as [x,y].

    Parameters
    ----------
    point : dict
        The metadata point to convert

    Returns
    -------
    np.ndarray
        Specified metadata point converted to numpy array.
    """
    return np.array([point["x"], point["y"]])