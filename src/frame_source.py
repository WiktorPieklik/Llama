import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Optional
from typing import Union

import numpy as np
from cv2 import cv2


class FrameSource(ABC):
    """ Abstract frame source base class. """

    @abstractmethod
    def get_frame(self) -> np.ndarray:
        """ Returns frame from this image source. """
        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        return self.get_frame()


class ThreadedFrameSource(FrameSource, Thread, ABC):
    def __init__(self):
        super().__init__()
        self.frame_queue = Queue()
        self.is_join_requested = False

    def get_frame(self, block: bool = True) -> np.ndarray:
        """ Returns oldest frame from `self.frame_queue`.

        Parameters
        ----------
        block : bool, optional
            Defines blocking behavior of getting element from queue.

        Returns
        -------
        :obj:`np.ndarray`
            Oldest frame from `self.frame_queue`.
        """
        return self.frame_queue.get(block=block)

    @abstractmethod
    def run(self) -> None:
        super().run()

    def join(self, timeout: Optional[float] = ...) -> None:
        """ Attempts to join this thread.

        Parameters
        ----------
        timeout : float, optional
            Join timeout duration in seconds.
        """
        self.is_join_requested = True
        super().join(timeout)


class ImageFrameSource(FrameSource):
    """ Frame source wrapper for :class:`cv2.VideoCapture` class. """

    def __init__(self, image_path: Union[str, Path], flags: int = cv2.IMREAD_COLOR):
        """
        Parameters
        ----------
        image_path : str or :obj:Path
            Path to target image.
        flags : int
            Image read flags.

        See Also
        --------
        :func:`cv2.imread`
        """
        if not os.path.exists(image_path):
            raise ValueError("Specified image file doesn't exist")
        self.frame = cv2.imread(filename=image_path, flags=flags)

    def get_frame(self) -> np.ndarray:
        return self.frame.copy()


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
        ret, frame = self.capture_device.read()
        return frame


class VideoFrameSource(ThreadedFrameSource):
    """ Frame source for loading video files. """

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
        if not os.path.exists(video_path):
            raise ValueError("The specified video file doesn't exist.")
        self.capture_device = cv2.VideoCapture(video_path)
        self.fps = self.capture_device.get(cv2.CAP_PROP_FPS)
        self.frame_duration = 1 / self.fps

    def run(self) -> None:
        """ Reads frames from `self.capture_device` until thread join is requested.

        After reading each frame, waits for remaining frame duration time to
        maintain corresponding playback speed.
        """
        while not self.is_join_requested:
            read_start = time.time()
            ret, frame = self.capture_device.read()
            self.frame_queue.put(frame)
            time.sleep(time.time() - read_start)
