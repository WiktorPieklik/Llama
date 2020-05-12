from pathlib import Path

import dlib
import numpy as np


class ShapePredictor:
    def __init__(self, model_path: str):
        """
        Parameters
        ----------
        model_path : str
            Path to the trained shape prediction model.
        """
        abs_model_path = str(Path(model_path).resolve())
        self._predictor: dlib.shape_predictor = dlib.shape_predictor(abs_model_path)

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
