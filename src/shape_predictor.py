from typing import Dict
from pathlib import Path

import dlib
import numpy as np
import os
from config import PREDICTOR_LANDMARKS


class ShapePredictor:
    def __init__(self, model_path: str):
        """
        Parameters
        ----------
        model_path : str
            Path to the trained shape prediction model.
        """
        if not os.path.exists(model_path):
            raise ValueError("Specified model file doesn't exist.")
        self._predictor: dlib.shape_predictor = dlib.shape_predictor(model_path)

    def predict(
        self, im_gray: np.ndarray, face: dlib.rectangle
    ) -> dlib.full_object_detection:
        """ Returns shape predicted from the specified image.

        Parameters
        ----------
        im_gray : np.ndarray
            Input image in grayscale.
        face : dlib.rectangle
            Area containing a detected face.

        Returns
        -------
        dlib.full_object_detection
        """
        return self._predictor(im_gray, face)

    def predict_remapped(
        self, im_gray: np.ndarray, face: dlib.rectangle
    ) -> Dict[int, dlib.point]:
        return remap_points(self.predict(im_gray, face))


def remap_points(detected_points: dlib.full_object_detection) -> dict:
    remapped_points = {}
    for i, part in enumerate(detected_points.parts()):
        remapped_points[PREDICTOR_LANDMARKS[i]] = part
    return remapped_points
