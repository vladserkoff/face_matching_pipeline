""" Code adopted from Tensorflow implementation of the face detection algorithm found at
https://github.com/davidsandberg/facenet/blob/master/src/align/align_dataset_mtcnn.py
"""

from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from secondface.mtcnn.detector import MTCNN

MINSIZE = 50  # minimum size of a face
THRESHOLDS = (0.6, 0.7, 0.95)  # three steps's thresholds
FACTOR = 0.709  # scale factor sqrt(0.5)
MARGIN = 0.1
CROPPED_FACE_SIZE = 160


class FaceDetector:
    """
    Detect and align faces on an image.
    """

    def __init__(self,
                 min_face_size: int = MINSIZE,
                 thresholds: Tuple[float, float, float] = THRESHOLDS,
                 margin: float = MARGIN) -> None:
        """
        Parameters
        ----------
        min_face_size : int
            Don't detect faces less than `min_face_size` along any axis.
        thresholds : list or tuple
            3 thresholds for 3 stages of MTCNN face detection algorithm
        margin : float
            Margin (in percents) around a detected face. Needed for facenet.
        """
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.margin = margin
        self._detector = MTCNN(self.min_face_size, self.thresholds)

    def find_faces(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Detects faces in an image, and returns bounding boxes,
        cropped faces and model's confidence in the image being a face.

        Parameters
        ----------
        image : `PIL.Image.Image` instance

        Returns
        -------
        detected_faces : list of dicts
            Each dictionary contains following elements:
                `box`: list, [x_left, y_top, width, height] of the face area
                `confidence`: float, model's confidence  in the image being a face
                `face`: `PIL.Image.Image`, cropped face, resized for facenet model
        """
        bounding_boxes = self._detector.detect_faces(image)

        detected_faces = []
        for box in bounding_boxes:
            x_left, y_top, x_right, y_bottom, confidence = box
            width = x_right - x_left
            height = y_bottom - y_top
            crop_points = np.zeros(4, dtype=np.int32)
            crop_points[0] = np.maximum(x_left - width * self.margin / 2, 0)
            crop_points[1] = np.maximum(y_top - height * self.margin / 2, 0)
            crop_points[2] = np.minimum(x_right + width * self.margin / 2,
                                        image.size[0])
            crop_points[3] = np.minimum(y_bottom + height * self.margin / 2,
                                        image.size[1])
            face = image.crop(crop_points)
            face = face.resize(
                (CROPPED_FACE_SIZE, CROPPED_FACE_SIZE),
                resample=Image.BILINEAR)
            face_data = {
                'box': crop_points.tolist(),
                'face': face,
                'confidence': confidence
            }
            detected_faces.append(face_data)
        return detected_faces

    @staticmethod
    def plot_boxes(image: Image.Image,
                   detected_faces: List[Dict[str, Any]]) -> Image.Image:
        """
        Plot original image with boxes around identified faces.

        Parameters
        ----------
        image : `PIL.Image.Image` instance
            Image used in `find_faces` step
        detected_faces : list of dicts
            Dictionaries with `box` fields for every face found on the `image`

        Returns
        -------
        image_with_boxes : `PIL.Image.Image`
            Original image with boxes drawn around identified faces.
        """
        image = image.copy()
        draw = ImageDraw.Draw(image)
        for face in detected_faces:
            draw.rectangle(face['box'])
        return image
