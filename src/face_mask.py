from abc import ABC, abstractmethod
from typing import Tuple

import dlib
import numpy as np
from cv2 import cv2


class FaceMaskGeneric(ABC):
    """ Generic class for applying a face mask. """

    def __init__(self):
        pass

    @abstractmethod
    def apply(
        self, input_img: np.ndarray, face_landmarks: dlib.full_object_detection
    ) -> None:
        """ Returns image with this mask applied on the input image.

        Parameters
        ----------
        input_img : :obj:`np.ndarray`
            Input image
        face_landmarks : :obj:`dlib.full_object_detection`

        Returns
        -------
        :obj:`np.ndarray`
            Image with applied mask
        """
        pass


class FaceMaskPoints(FaceMaskGeneric):
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
