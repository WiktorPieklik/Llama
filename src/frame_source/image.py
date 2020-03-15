from pathlib import Path
from typing import Union

import numpy as np
from cv2 import cv2

from .source import FrameSource


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
        self.frame = cv2.imread(filename=image_path, flags=flags)

    def get_frame(self) -> np.ndarray:
        return self.frame.copy()
