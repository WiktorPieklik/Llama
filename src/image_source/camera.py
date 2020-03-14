from cv2 import cv2

from .source import ThreadedFrameSource


class CameraFrameSource(ThreadedFrameSource):
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

    def run(self) -> None:
        """ Reads frames from `self.capture_device` until thread join is requested. """
        while not self.is_join_requested:
            ret, frame = self.capture_device.read()
            self.frame_queue.put(frame)
