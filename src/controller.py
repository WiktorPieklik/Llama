from threading import Thread
from typing import Optional

from cv2 import cv2

from config import SHAPE_PREDICTOR_MODEL_PATH
from face_detector import FaceDetector
from face_mask.mask import FaceMaskHaircut, FaceMaskEyes, FaceMaskMoustache, MaskFactory
from frame_source import CameraFrameSource, FrameSource, ThreadedFrameSource
from shape_predictor import ShapePredictor


class Controller(Thread):
    def __init__(self):
        super(Controller, self).__init__()
        self._is_join_requested = False

        self._face_detector = FaceDetector()
        self._shape_predictor = ShapePredictor(model_path=SHAPE_PREDICTOR_MODEL_PATH)
        self._frame_source = None
        # self.frame_source = ImageFrameSource("src/keanu.png")
        # self.frame_source = VideoFrameSource("data/external/pexels_video/1.mp4")
        self.frame_source = CameraFrameSource(0)
        # self._next_frame_source: FrameSource = None
        # self._ui = UI(self)
        # self._mask = FaceMaskPoints(point_radius=1, point_color=(255, 255, 0))
        self._mask = MaskFactory().create_mask("src/face_mask/assets/masks/eyes/hehe.png")

    def join(self, timeout: Optional[float] = ...) -> None:
        """ Attempts to join this thread.

        Parameters
        ----------
        timeout : float, optional
            Join timeout duration in seconds.
        """
        self._is_join_requested = True
        super().join(timeout)

    def run(self):
        while not self._is_join_requested:
            frame_color = self._frame_source.get_frame()
            frame_gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
            faces = self._face_detector.detect(frame_gray)

            for face in faces:
                face_points = self._shape_predictor.predict_remapped(frame_gray, face)
                self._mask.apply(frame_color, face_points)

            cv2.imshow("Frame", frame_color)
            cv2.waitKey(1)

    @property
    def frame_source(self) -> FrameSource:
        return self._frame_source

    @frame_source.setter
    def frame_source(self, new_frame_source: FrameSource):
        if isinstance(self._frame_source, ThreadedFrameSource):
            # Stop current threaded frame source
            self._frame_source.join()
        self._frame_source = new_frame_source
        if isinstance(self._frame_source, ThreadedFrameSource):
            # Start new threaded frame source
            self._frame_source.start()

    def update_ui(self):
        raise NotImplementedError
