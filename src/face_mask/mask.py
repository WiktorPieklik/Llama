from abc import ABC, abstractmethod

import dlib
import numpy as np


class FaceMask(ABC):
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
