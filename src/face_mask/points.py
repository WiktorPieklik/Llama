from typing import Tuple

import dlib
import numpy as np
from cv2 import cv2

from .mask import FaceMask


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
        self, input_img: np.ndarray, landmarks: dlib.full_object_detection
    ) -> None:
        """ Draws points over face landmarks.

        Parameters
        ----------
        input_img : :obj:`np.ndarray`
            Input frame
        landmarks : :obj:`dlib.full_object_detection`
            Detected face landmarks.
        """
        for i in range(landmarks.num_parts):
            current_landmark = landmarks.part(i)
            cv2.circle(
                input_img,
                (current_landmark.x, current_landmark.y),
                self._point_radius,
                self._point_color,
                -1,
            )
