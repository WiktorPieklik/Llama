import dlib
import numpy as np


class FaceDetector:
    """ Class for detecting areas containg faces. """

    def __init__(self):
        self._detector = dlib.get_frontal_face_detector()

    def detect(self, input_frame: np.ndarray) -> dlib.rectangles:
        """ Returns detected faces """
        return self._detector(input_frame)
