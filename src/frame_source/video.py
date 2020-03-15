import queue
import time
from pathlib import Path
from queue import Queue
from typing import Union

import numpy as np
from cv2 import cv2

from .source import ThreadedFrameSource


class VideoFrameSource(ThreadedFrameSource):
    """ Frame source for loading video files. """

    FRAME_BUFFER_TIME_SPAN = 2
    FRAME_BUFFER_RETRIEVE_TIMEOUT = 0.2

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

        frame_buffer_size = self.fps * self.FRAME_BUFFER_TIME_SPAN
        self.frame_buffer = Queue(frame_buffer_size)

        self.time_last_frame_get = time.time()
        self.frame_get_count = 0

    def __next__(self) -> np.ndarray:
        try:
            return self.get_frame()
        except queue.Empty:
            raise StopIteration

    def get_frame(self, timeout=FRAME_BUFFER_RETRIEVE_TIMEOUT) -> np.ndarray:
        """ Returns frame from this image source. """
        while time.time() - self.time_last_frame_get < self.frame_duration:
            pass

        self.time_last_frame_get = time.time()
        print("Returned frame: {}".format(self.frame_get_count))
        self.frame_get_count += 1

        return self.frame_buffer.get(block=True, timeout=timeout)

    def run(self) -> None:
        """ Reads frames from `self.capture_device` until thread join is requested.

        After reading each frame, waits for remaining frame duration time to
        maintain corresponding playback speed.
        """
        frame_loaded_count = 0
        while not self.is_join_requested:
            if not self.frame_buffer.full():
                is_frame_ok, frame = self.capture_device.read()
                if not is_frame_ok:
                    # All video frames loaded
                    break

                self.frame_buffer.put(frame)
                frame_loaded_count += 1
                print("Buffered frame: {}".format(frame_loaded_count))
