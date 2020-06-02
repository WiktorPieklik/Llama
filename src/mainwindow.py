import sys
from typing import Type

import numpy as np
from PyQt5 import QtWidgets
from PyQt5 import uic
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QPixmap, QImage
from cv2 import cv2

import utils
from config import SHAPE_PREDICTOR_MODEL_PATH
from face_detector import FaceDetector
from face_mask.mask import FaceMaskDrawPoints, FaceMask
from frame_source import (
    FrameSource,
    ThreadedFrameSource,
    CameraFrameSource,
)
from shape_predictor import ShapePredictor

MainWindowUi, QtBaseClass = uic.loadUiType(
    utils.get_path_relative_to_caller("ui/mainwindow.ui")
)


class ImageProcessingThread(QThread):
    send_image = pyqtSignal(QImage)

    def __init__(self, parent):
        super().__init__(parent)
        self._is_join_requested = False
        self._frame_source = None
        self._mask = None

        self._face_detector = FaceDetector()
        self._shape_predictor = ShapePredictor(model_path=SHAPE_PREDICTOR_MODEL_PATH)

        self.mask = FaceMaskDrawPoints(point_radius=1, point_color=(255, 255, 0))
        self.frame_source = CameraFrameSource(0)

    def run(self):
        while not self._is_join_requested:
            frame_color = self._frame_source.get_frame()
            frame_gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
            faces = self._face_detector.detect(frame_gray)

            for face in faces:
                face_points = self._shape_predictor.predict_remapped(frame_gray, face)
                self._mask.apply(frame_color, face_points)

            height, width, channel_count = frame_color.shape
            bytes_per_line = channel_count * width
            image_qt = QImage(
                frame_color.data, width, height, bytes_per_line, QImage.Format_BGR888,
            )
            self.send_image.emit(image_qt)

    @property
    def frame_source(self) -> Type[FrameSource]:
        return self._frame_source

    @frame_source.setter
    def frame_source(self, new_frame_source: Type[FrameSource]):
        if isinstance(self._frame_source, ThreadedFrameSource):
            # Stop current threaded frame source
            self._frame_source.join()
        self._frame_source = new_frame_source
        if isinstance(self._frame_source, ThreadedFrameSource):
            # Start new threaded frame source
            self._frame_source.start()

    @property
    def mask(self) -> FaceMask:
        return self._mask

    @mask.setter
    def mask(self, new_mask: Type[FaceMask]):
        self._mask = new_mask


class MainWindow(QtWidgets.QMainWindow, MainWindowUi):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        MainWindowUi.__init__(self)
        self.setupUi(self)
        thread_image_proc = ImageProcessingThread(self)
        thread_image_proc.send_image.connect(self.set_image)
        thread_image_proc.start()

    @pyqtSlot(QImage)
    def set_image(self, image_qt: QImage):
        self.image_display.setPixmap(QPixmap.fromImage(image_qt))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    app.exec_()
