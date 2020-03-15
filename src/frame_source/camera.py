import numpy as np
from cv2 import cv2

from .source import FrameSource


class CameraFrameSource(FrameSource):
    """ Frame source wrapper for :class:`cv2.VideoCapture` class. """

    def __init__(self, video_device_id: int = 0):
        """
        Parameters
        ----------
        video_device_id : int, optional
            Id of the target video device.
        """
        super().__init__()
        self.capture_device = cv2.VideoCapture(video_device_id)

    def get_frame(self) -> np.ndarray:
        return self.capture_device.read()[1]
