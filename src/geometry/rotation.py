from cv2 import cv2
import numpy as np


class RotatorNonCropping:
    """ Class for performing non-cropping image rotation.

    Single use only.
    """

    def __init__(self, image_input: np.ndarray, angle: float):
        """
        Parameters
        ----------
        image_input : np.ndarray
            Image to rotate
        angle : float
            Rotation angle, expressed in degrees.
        """
        self._image_input = image_input
        self._angle = angle

        # Cached values
        self._image_input_center = None
        self._image_rotated = None
        self._image_rotated_bounds = None
        self._shift_after_rotation = None
        self._rotation_mat_default = None

    @property
    def image_input(self) -> np.ndarray:
        return self._image_input

    @property
    def angle(self) -> float:
        return self._angle

    @property
    def image_input_center(self) -> tuple:
        if self._image_input_center is None:
            height, width = self.image_input.shape[:2]  # Image shape has 3 dimensions
            self._image_input_center = (width / 2, height / 2)
        return self._image_input_center

    @property
    def image_rotated_bounds(self) -> tuple:
        if self._image_rotated_bounds is None:
            result = self.rotation_mat_default
            # rotation calculates the cos and sin, taking absolutes of those.
            abs_cos = abs(result[0, 0])
            abs_sin = abs(result[0, 1])
            # Find the new width and height bounds
            height, width = self.image_input.shape[:2]  # Image shape has 3 dimensions
            bound_w = int(height * abs_sin + width * abs_cos)
            bound_h = int(height * abs_cos + width * abs_sin)

            self._image_rotated_bounds = (bound_w, bound_h)
        return self._image_rotated_bounds

    @property
    def rotation_mat_default(self) -> np.ndarray:
        if self._rotation_mat_default is None:
            self._rotation_mat_default = cv2.getRotationMatrix2D(
                self.image_input_center, self.angle, 1.0
            )
        return self._rotation_mat_default

    @property
    def rotation_mat_shifted(self) -> np.ndarray:
        result = np.copy(self.rotation_mat_default)
        # subtract input image center (bringing image back to origin) and
        # add the rotated image center coordinates.
        result[0, 2] += self.shift_after_rotation[0]
        result[1, 2] += self.shift_after_rotation[1]
        return result

    @property
    def shift_after_rotation(self) -> np.ndarray:
        if self._shift_after_rotation is None:
            self._shift_after_rotation = np.array(
                (
                    self.image_rotated_bounds[0] / 2 - self.image_input_center[0],
                    self.image_rotated_bounds[1] / 2 - self.image_input_center[1],
                ),
                dtype=int,
            )
        return self._shift_after_rotation

    @property
    def image_rotated(self) -> np.ndarray:
        if self._image_rotated is None:
            # rotate image with the new bounds and translated rotation matrix
            self._image_rotated = cv2.warpAffine(
                self.image_input, self.rotation_mat_shifted, self.image_rotated_bounds
            )
        return self._image_rotated

    def get_point_after_rotation(self, point: np.ndarray) -> np.ndarray:
        rotation_cos = self.rotation_mat_default[0, 0]  # cos
        rotation_sin = self.rotation_mat_default[0, 1]  # sin

        pivot = np.array([self.image_input.shape[1], self.image_input.shape[0]]) / 2

        point_shifted_by_pivot = point - pivot
        rotated_x = (
            point_shifted_by_pivot[0] * rotation_cos
            + point_shifted_by_pivot[1] * rotation_sin
        )
        rotated_y = (
            - point_shifted_by_pivot[0] * rotation_sin
            + point_shifted_by_pivot[1] * rotation_cos
        )

        result = (
            np.array([rotated_x, rotated_y], dtype=int)
            + pivot
            + self.shift_after_rotation
        ).astype(np.int)
        return result
