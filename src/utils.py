import math
import numpy as np
import json
import dlib


def load_json(json_path: str) -> dict:
    with open(json_path, "r") as json_file:
        return json.load(json_file)


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
