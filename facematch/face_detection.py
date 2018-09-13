# pylint: disable = R0914
""" Code adopted from Tensorflow implementation of the face detection algorithm found at
https://github.com/davidsandberg/facenet/blob/master/src/align/align_dataset_mtcnn.py
"""

import os
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image

from facematch.face import Face
from facematch.faceboxes.faceboxes import FaceBoxes
from facematch.viz import resize

MIN_CONFIDENCE = 0.9
MARGIN = 0.1
CROPPED_FACE_SIZE = 160  # as expected by facenet
RESIZE_TO = (1024, 1024)


class FaceDetector:
    """
    Detect and align faces on an image.
    """

    def __init__(self,
                 min_confidence: float = MIN_CONFIDENCE,
                 margin: float = MARGIN) -> None:
        """
        Parameters
        ----------
        min_confidence : float
            Minimum confidence that given zone contains a face.
        margin : float
            Margin (in percents) around a detected face. Needed for facenet.
        """
        self.min_confidence = min_confidence
        self.model_path = os.path.join(
            os.path.dirname(__file__), 'faceboxes', 'model', 'faceboxes-tf.pb')
        self.faceboxes = FaceBoxes(self.model_path)
        self.faceboxes.load_model()
        self.margin = margin

    def find_faces(self,
                   image: Image.Image,
                   resize_to: Tuple[int, int] = RESIZE_TO) -> List[Face]:
        """
        Detects faces in an image, and returns bounding boxes,
        cropped faces and model's confidence in the image being a face.

        Parameters
        ----------
        image : `PIL.Image.Image`
        resize_to : tuple of length 2, default (1024, 1024)
            If an image is large, it's advisable to shrink it into 1024x1024 box.

        Returns
        -------
        detected_faces : list of `facematch.face.Face`s
            Each `Face` object contains following elements:
                `box`: list, [x_left, y_top, x_right, y_bottom] of the face area
                `confidence`: float, model's confidence in the image being a face
                `image`: `PIL.Image.Image`, cropped face, resized for facenet model
        """
        img_width, img_hight = image.size
        if resize_to:
            assert len(resize_to) == 2
            resized = resize(image, resize_to)
            image_np = np.asarray(resized)
        else:
            image_np = np.asarray(image)

        image_np = np.expand_dims(image_np, 0)
        boxes, scores = self.faceboxes.forward(image_np)

        boxes = boxes[0]
        scores = scores[0]
        to_keep = scores > self.min_confidence
        boxes = boxes[to_keep]
        scores = scores[to_keep]
        scaler = np.array([img_hight, img_width, img_hight, img_width],
                          dtype='float32')
        boxes = boxes * scaler

        detected_faces = []
        for box, confidence in zip(boxes, scores):
            final_box = self._get_box_coordinates(img_width, img_hight, box)
            face = image.crop(final_box)
            face = face.resize((CROPPED_FACE_SIZE, CROPPED_FACE_SIZE),
                               resample=Image.BILINEAR)
            face_data = Face(box=final_box, image=face, confidence=confidence)
            detected_faces.append(face_data)

        detected_faces = sorted(detected_faces, key=lambda x: x.box[0])
        return detected_faces

    def _get_box_coordinates(self, img_width: int, img_hight: int,
                             original_box: List[float]) -> List[int]:
        y_top, x_left, y_bottom, x_right = original_box
        width = x_right - x_left
        height = y_bottom - y_top
        box = np.zeros(4)
        box[0] = np.maximum(x_left - width * self.margin / 2, 0)
        box[1] = np.maximum(y_top - height * self.margin / 2, 0)
        box[2] = np.minimum(x_right + width * self.margin / 2, img_width)
        box[3] = np.minimum(y_bottom + height * self.margin / 2, img_hight)
        box = box.round().astype(int).tolist()
        return box

    def find_faces_batch(
            self,
            images: Iterable[Image.Image],
            resize_to: Tuple[int, int] = RESIZE_TO) -> List[List[Face]]:
        """
        Find faces on multiple images

        Parameters
        ----------
        images : iterable of `PIL.Image.Image`s
        resize_to : tuple of length 2, default (1024, 1024)
            If images are large, it's advisable to shrink them into 1024x1024 box.

        Returns
        -------
        detected_faces : list
            List with length `len(images)`,
            each element is a list with detected faces for particular image.
        """
        detected_faces = []
        for i, image in enumerate(images):
            detected_faces.append(self.find_faces(image, resize_to))
            print('\r%i images processed' % (i + 1), end='')
        print()
        return detected_faces
