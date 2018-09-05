# pylint: disable = R0903
"""
Face embedding based on original Tensorflow implementation of Facenet
"""
import os
from typing import List

import numpy as np
from PIL import Image

from secondface.facenet.facenet import FaceNet


class FaceEncoder:
    """
    Given an image of a face, compute it's embedding.
    """

    def __init__(self) -> None:
        self.model_path = os.path.join(
            os.path.dirname(__file__), 'facenet', 'model',
            'facenet-vggface2.pb')
        self.facenet = FaceNet(self.model_path)
        self.facenet.load_model()

    def calculate_embeddings(self,
                             images: List[Image.Image],
                             flip: bool = True) -> np.ndarray:
        """
        Calculate face embeddings

        Parameters
        ----------
        images : list of `PIL.Image.Image`s
            Cropped and resized images of faces.
        flip : bool, default True
            Weather to flip images and return average embeddings
            for flipped and non-flipped variants.

        Returns
        -------
        embeddings : np.ndarray
            Each list contains vector representation of a face
        """
        images_np = np.stack([np.array(x) for x in images])
        if flip:
            images_flip = [x.transpose(Image.FLIP_LEFT_RIGHT) for x in images]
            flipped_np = np.stack([np.array(x) for x in images_flip])
            images_np = np.concatenate([images_np, flipped_np])
        images_np = (images_np - 127.5) / 128.0
        embeddings = self.facenet.forward(images_np)
        if flip:
            new_shape = (-1, len(images), embeddings.shape[-1])
            embeddings = embeddings.reshape(new_shape)
            embeddings = np.mean(embeddings, axis=0)
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings
