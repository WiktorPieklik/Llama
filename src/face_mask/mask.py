import enum
import os
from abc import ABC, abstractmethod

import dlib
import numpy as np
from typing import Dict, Tuple
from cv2 import cv2

from src import utils


class FaceMask(ABC):
    """ Generic class for applying a face mask. """

    def __init__(self):
        pass

    @abstractmethod
    def apply(
        self, input_img: np.ndarray, face_points: Dict[int, dlib.point]
    ) -> np.ndarray:
        """ Returns image with this mask applied on the input image.

        Parameters
        ----------
        input_img : :obj:`np.ndarray`
            Input image
        face_points : dict of int : dlib.point

        Returns
        -------
        :obj:`np.ndarray`
            Image with applied mask
        """
        pass


class FaceMaskPoints(FaceMask):
    """ FaceMask for drawing points over face landmarks. """

    def __init__(
        self, point_radius: int = 2, point_color: Tuple[int, int, int] = (255, 0, 0)
    ):
        """
        Parameters
        ----------
        point_radius : int
            Radius of points drawn over each face landmark.
        point_color : tuple of int
            Color of points drawn over each face landmark.
        """
        super().__init__()
        self._point_radius = point_radius
        self._point_color = point_color

    def apply(
        self, input_img: np.ndarray, face_points: Dict[int, dlib.point]
    ) -> np.ndarray:
        """ Draws points over face landmarks.

        Parameters
        ----------
        input_img : `np.ndarray`
            Input frame
        face_points : dict of int:`dlib.point`
            Detected face landmarks.
        """
        for point in face_points.values():
            cv2.circle(
                input_img,
                (point.x, point.y),
                self._point_radius,
                self._point_color,
                -1,
            )


class ImageAssetMixin:
    """ Mixin implementing image asset functionality. """

    def __init__(self, path_asset: str):
        if not os.path.exists(path_asset):
            raise ValueError("Specified asset file doesn't exist.")
        self._asset: np.ndarray = cv2.imread(path_asset, cv2.IMREAD_UNCHANGED)

        path_md_json = os.path.splitext(path_asset)[0] + ".json"
        self._metadata: dict = utils.load_json(path_md_json)

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def asset(self) -> np.ndarray:
        return self._asset.copy()


class FaceMaskHaircut(FaceMask, ImageAssetMixin):
    def __init__(self, path_asset_image: str):
        FaceMask.__init__(self)
        ImageAssetMixin.__init__(self, path_asset_image)

    def apply(self, input_img: np.ndarray, face_points: Dict[int, dlib.point]) -> None:

        points_source = np.float32(
            [list(self.metadata[key].values()) for key in ["0", "0", "16", "16"]]
        )
        # point_tl_source = get_top_left_point(points_source)
        points_target = np.float32(
            [[face_points[i].x, face_points[i].y] for i in [0, 0, 16, 16]]
        )
        # point_tl_target = get_top_left_point(points_target)

        transform_matrix = cv2.getPerspectiveTransform(points_source, points_target)
        points_source_warped = [
            get_point_warped(transform_matrix, p) for p in points_source
        ]
        transformed = transform_perspective(self.asset, transform_matrix)
        cv2.imshow("Transformed", transformed)
        cv2.waitKey(1)


def get_top_left_point(points):
    np.array(list(np.min(points[:, i]) for i in range(2)))


def transform_perspective(image: np.ndarray, perspective_transform):
    image_warped = cv2.warpPerspective(image, perspective_transform, (1000, 1000))
    return image_warped


def get_point_warped(perspective_transform, point):
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
