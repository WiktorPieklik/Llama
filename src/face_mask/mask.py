import os
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Dict, Tuple

import dlib
import numpy as np
from cv2 import cv2

from geometry import point as utils_point
from geometry import vector as utils_vector
from geometry.misc import overlay_transparent
from geometry.rotation import RotatorNonCropping
from src import utils


class FaceMask(ABC):
    """ Generic class for applying a face mask. """

    def __init__(self):
        pass

    @abstractmethod
    def apply(
        self, input_img: np.ndarray, face_points: Dict[int, dlib.point]
    ) -> np.ndarray:
        """ Returns image with this mask applied on the input image.

        Parameters
        ----------
        input_img : :obj:`np.ndarray`
            Input image
        face_points : dict of int : dlib.point

        Returns
        -------
        :obj:`np.ndarray`
            Image with applied mask
        """
        pass


class FaceMaskDrawPoints(FaceMask):
    """ FaceMask for drawing points over face landmarks. """

    def __init__(
        self, point_radius: int = 2, point_color: Tuple[int, int, int] = (255, 0, 0)
    ):
        """
        Parameters
        ----------
        point_radius : int
            Radius of points drawn over each face landmark.
        point_color : tuple of int
            Color of points drawn over each face landmark.
        """
        super().__init__()
        self._point_radius = point_radius
        self._point_color = point_color

    def apply(
        self, input_img: np.ndarray, face_points: Dict[int, dlib.point]
    ) -> np.ndarray:
        """ Draws points over face landmarks.

        Parameters
        ----------
        input_img : `np.ndarray`
            Input frame
        face_points : dict of int:`dlib.point`
            Detected face landmarks.
        """
        for point in face_points.values():
            cv2.circle(
                input_img,
                (point.x, point.y),
                self._point_radius,
                self._point_color,
                -1,
            )
        return input_img


class ImageAssetMixin:
    """ Mixin implementing image asset functionality. """

    def __init__(self, path_asset: str):
        if not os.path.exists(path_asset):
            raise ValueError("Specified asset file doesn't exist.")
        self._asset: np.ndarray = cv2.imread(path_asset, cv2.IMREAD_UNCHANGED)

        path_md_json = os.path.splitext(path_asset)[0] + ".json"

        if not os.path.exists(path_md_json):
            raise ValueError(
                "Matching asset metadata doesn't exist. Expected file: '{}'.".format(
                    path_md_json
                )
            )
        self._metadata: dict = utils.json_file_to_dict(path_md_json)

    @property
    def metadata(self) -> dict:
        """ Returns metadata structure.

        Returns
        -------
        dict
            This asset's metadata structure.
        """
        return self._metadata

    @property
    def asset(self) -> np.ndarray:
        """ Returns copy of asset.

        Returns
        -------
        np.ndarray
            Copy of this asset's image.
        """
        return self._asset.copy()


class FaceMaskTwoPointAssetAlignment(FaceMask, ImageAssetMixin):
    """ Generic class for aligning an asset onto a face, based on two pairs of
    reference points.
    """

    def __init__(self, path_asset_image: str):
        FaceMask.__init__(self)
        ImageAssetMixin.__init__(self, path_asset_image)

    @abstractmethod
    def get_ref_points_face(self, face_points: Dict[int, dlib.point]) -> np.ndarray:
        """ Returns reference points detected on face.

        The asset is aligned to the reference points detected on a face.
        """
        pass

    @abstractmethod
    def get_ref_points_asset(self) -> np.ndarray:
        """ Returns reference points on asset.

        Based on these points, the asset is aligned to the reference points
        detected on a face.
        """
        pass

    def apply(
        self, image_input: np.ndarray, face_points: Dict[int, dlib.point]
    ) -> np.ndarray:
        # Get points necessary for geometric transformations
        vector_bound_asset = self.get_ref_points_asset()
        vector_free_asset = utils_vector.convert_bound_to_free(vector_bound_asset)
        vector_bound_face = self.get_ref_points_face(face_points=face_points)
        vector_free_face = utils_vector.convert_bound_to_free(vector_bound_face)

        # Rotate asset
        angle_asset_rotation = utils_vector.calc_angle_clockwise(
            vector_free_asset, vector_free_face
        )
        rotator = RotatorNonCropping(self.asset, angle=angle_asset_rotation)

        # Resize image
        scale_factor = np.linalg.norm(vector_free_face) / np.linalg.norm(
            vector_free_asset
        )
        asset_size_scaled = tuple(
            int(val * scale_factor) for val in rotator.image_rotated.shape[:2][::-1]
        )
        asset_rotated_scaled = cv2.resize(
            src=rotator.image_rotated, dsize=asset_size_scaled
        )

        # Calculate overlay coordinates
        point_ref_input = vector_bound_face[0]
        point_ref_asset = (
            scale_factor * rotator.get_point_after_rotation(point=vector_bound_asset[0])
        ).astype(np.int)
        position_asset = point_ref_input - point_ref_asset
        result = overlay_transparent(image_input, asset_rotated_scaled, position_asset)
        return result


class FaceMaskHaircut(FaceMaskTwoPointAssetAlignment):
    def __init__(self, path_asset_image: str):
        FaceMaskTwoPointAssetAlignment.__init__(self, path_asset_image)

    def get_ref_points_face(self, face_points: Dict[int, dlib.point]) -> np.ndarray:
        return np.array(
            [utils_point.dlib_point_to_np_array(face_points[key]) for key in [0, 16]],
            dtype=np.int,
        )

    @lru_cache(maxsize=1)
    def get_ref_points_asset(self) -> np.ndarray:
        return np.array(
            [
                utils_point.point.md_point_to_np_array(self.metadata[key])
                for key in ["0", "16"]
            ],
            dtype=np.int,
        )


class FaceMaskEyes(FaceMaskTwoPointAssetAlignment):
    """ FaceMask for aligning an asset, based on center points of eyes. """

    def get_ref_points_face(self, face_points: Dict[int, dlib.point]) -> np.ndarray:
        ref_points_face = np.array(
            [
                np.array(
                    [
                        utils_point.dlib_point_to_np_array(
                            face_points[points_eye_corners[0]]
                        ),
                        utils_point.dlib_point_to_np_array(
                            face_points[points_eye_corners[1]]
                        ),
                    ]
                ).mean(axis=0)
                for points_eye_corners in [[36, 39], [45, 42]]
            ],
            dtype=np.int,
        )
        return ref_points_face

    @lru_cache(maxsize=1)
    def get_ref_points_asset(self) -> np.ndarray:
        return np.array(
            [
                [point["x"], point["y"]]
                for point in [
                    self.metadata["eye_center_left"],
                    self.metadata["eye_center_right"],
                ]
            ],
            dtype=np.int,
        )
