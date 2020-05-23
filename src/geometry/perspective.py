import numpy as np
from cv2 import cv2


def transform_perspective(image: np.ndarray, perspective_transform):
    image_warped = cv2.warpPerspective(image, perspective_transform, (1000, 1000))
    return image_warped


def get_point_warped(perspective_transform: np.ndarray, point: np.ndarray):
    """ Returns location of point after applying perspective_transform.

    Parameters
    ----------
    perspective_transform :
        Perspective transform matrix.
    point :
        Input point to warp.

    Returns
    -------
    point_warped :
        Point after warping using the specified perspective transform.
    """
    px = (
        perspective_transform[0][0] * point[0]
        + perspective_transform[0][1] * point[1]
        + perspective_transform[0][2]
    ) / (
        (
            perspective_transform[2][0] * point[0]
            + perspective_transform[2][1] * point[1]
            + perspective_transform[2][2]
        )
    )
    py = (
        perspective_transform[1][0] * point[0]
        + perspective_transform[1][1] * point[1]
        + perspective_transform[1][2]
    ) / (
        (
            perspective_transform[2][0] * point[0]
            + perspective_transform[2][1] * point[1]
            + perspective_transform[2][2]
        )
    )

    point_warped = (int(px), int(py))
    return point_warped
