import sys
from pathlib import Path
from typing import Type, Any, List

from PyQt5 import QtWidgets, QtCore
from PyQt5 import uic
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt, QModelIndex
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QFileDialog
from cv2 import cv2

import utils
from config import SHAPE_PREDICTOR_MODEL_PATH
from face_detector import FaceDetector
from face_mask.mask import FaceMaskDrawPoints, FaceMask, MaskFactory, FaceMaskPassthrough
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
    signal_send_image = pyqtSignal(QImage)

    def __init__(self, parent):
        super().__init__(parent)
        self._is_join_requested = False
        self._is_joined = False

        self._frame_source = None
        self._mask = None
        self._mask_next = None

        self._face_detector = FaceDetector()
        self._shape_predictor = ShapePredictor(model_path=SHAPE_PREDICTOR_MODEL_PATH)

        self.mask = FaceMaskPassthrough()
        self.frame_source = CameraFrameSource(0)

    def run(self):
        while not self._is_join_requested:
            self.update_mask()

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
            self.signal_send_image.emit(image_qt)

        self._is_joined = True

    def join(self):
        self._is_join_requested = True
        while not self._is_joined:
            continue

    @property
    def frame_source(self) -> Type[FrameSource]:
        return self._frame_source

    @frame_source.setter
    def frame_source(self, new_frame_source: Type[FrameSource]):
        """ Sets the current frame source to the specified `new_frame_source`.

        If the current frame source is an instance of `ThreadedFrameSource`, then
        it is stopped before replacement.

        If `new_frame_source` if an instance of `ThreadedFrameSource`, then it is
        started after writing it to the `_frame_source` field.

        Parameters
        ----------
        new_frame_source : FrameSource
            New frame source.
        """
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
    def mask(self, mask_new: Type[FaceMask]):
        self._mask = mask_new

    def update_mask(self):
        """ Updates current mask with the next mask, if available. """
        if self._mask_next:
            self.mask = self._mask_next
            self._mask_next = None

    def handle_mask_receive(self, mask_next: Type[FaceMask]):
        """ Handles signal with the next mask to use.

        Parameters
        ----------
        mask_next : FaceMask
            Next mask to use.
        """
        self._mask_next = mask_next


class MaskListModel(QtCore.QAbstractListModel):
    """

    """

    def __init__(self, *args, masks=List[FaceMask], **kwargs):
        super().__init__(*args, **kwargs)
        self.masks = masks or []

    def data(self, index: QModelIndex, role: int = ...) -> Any:
        if role == Qt.DisplayRole:
            mask = self.masks[index.row()]
            return str(mask)

    def rowCount(self, parent: QModelIndex = ...) -> int:
        return len(self.masks)


class MainWindow(QtWidgets.QMainWindow, MainWindowUi):
    signal_send_mask = pyqtSignal(FaceMask)

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        MainWindowUi.__init__(self)
        self.setupUi(self)

        masks_dir_path = Path(
            utils.get_path_relative_to_caller("face_mask/assets/masks")
        )
        masks: List[FaceMask] = [
            MaskFactory.create_mask(str(fpath.resolve()))
            for fpath in masks_dir_path.rglob("*.png")
        ]
        masks.append(FaceMaskDrawPoints(point_radius=3, point_color=(255, 0, 0)))
        masks.sort(key=lambda mask: mask.name)
        masks.insert(0, FaceMaskPassthrough())
        self.face_mask_list_model = MaskListModel(masks=masks)
        self.face_mask_list.setModel(self.face_mask_list_model)

        self.thread_image_proc = ImageProcessingThread(self)
        self.thread_image_proc.signal_send_image.connect(self.handle_image_receive)

        self.signal_send_mask.connect(self.thread_image_proc.handle_mask_receive)
        self.face_mask_list.selectionModel().selectionChanged.connect(
            self.handle_face_mask_send
        )
        self.button_camera.clicked.connect(self.save_displayed_image_to_disk)

        self.thread_image_proc.start()

    def closeEvent(self, event):
        """ Called on PyQt close event. """
        self.thread_image_proc.join()

    def handle_face_mask_send(self):
        """ Sends the selected face mask instance to the image processing thread. """
        selected_index = self.face_mask_list.selectedIndexes()[0].row()
        selected_face_mask = self.face_mask_list_model.masks[selected_index]
        self.signal_send_mask.emit(selected_face_mask)

    # Signal slots
    @pyqtSlot(QImage)
    def handle_image_receive(self, image_qt: QImage):
        """ Updates the UI image display with the received `image_qt`.

        Parameters
        ----------
        image_qt : QImage
            Received image.
        """
        self.image_display.setPixmap(QPixmap.fromImage(image_qt))

    def save_displayed_image_to_disk(self):
        """ Attempts to save currently displayed image to disk.

        Returns
        -------
        True
            If image save operation was successful.
        False
            If image save failed.
        """
        pixmap = self.image_display.pixmap()
        if pixmap:
            filename = self.image_save_file_dialog()
            if filename:
                pixmap.save(filename, format="PNG")
                return True
        return False

    def image_save_file_dialog(self):
        """ Returns file path specified using a save file dialog.

        Returns
        -------
        str or None
            The specified file path.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filepath, _ = QFileDialog.getSaveFileName(
            self,  # Sets MainWindow as parent to make the dialog modal.
            "QFileDialog.getSaveFileName()",
            "",
            "PNG files (*.png)",
            options=options,
        )
        return filepath


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setDesktopSettingsAware(False)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    app.exec_()
