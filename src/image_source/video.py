import time
from pathlib import Path
from typing import Union

from cv2 import cv2

from .source import ThreadedFrameSource


class VideoFrameSource(ThreadedFrameSource):
    """ Frame source for loading video files. """

    NANOS_IN_SEC = 10 ** 9

    def __init__(self, video_path: Union[str, Path]):
        """
        Parameters
        ----------
        video_path : str or Path
            Path to the target video
        """
        super().__init__()
        if isinstance(video_path, Path):
            video_path = str(video_path.resolve())
        self.capture_device = cv2.VideoCapture(video_path)
        self.fps = self.capture_device.get(cv2.CAP_PROP_FPS)
        self.frame_duration = 1 / self.fps

    def run(self) -> None:
        """ Reads frames from `self.capture_device` until thread join is requested.

        After reading each frame, waits for remaining frame duration time to
        maintain corresponding playback speed.
        """
        while not self.is_join_requested:
            read_start = time.time_ns()
            ret, frame = self.capture_device.read()
            self.frame_queue.put(frame)
            read_end = time.time_ns()
            time.sleep((read_start - read_end) / self.NANOS_IN_SEC)
