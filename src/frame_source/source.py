from abc import ABC, abstractmethod
from queue import Queue
from threading import Thread
from typing import Optional

import numpy as np


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
        self.is_join_requested = False

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
