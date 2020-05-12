from abc import ABC
import numpy as np


class FaceDetector(ABC):
    def __init__(self):
        pass

    def detect(self, input_frame: np.ndarray):
        """ Returns detected faces or shapes"""
        raise NotImplementedError
