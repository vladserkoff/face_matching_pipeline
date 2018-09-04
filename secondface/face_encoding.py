"""
Face embedding based on original Tensorflow implementation of Facenet
"""
import os
from typing import List

import numpy as np
import torch
from PIL import Image

from secondface.facenet.facenet import FaceNet


class FaceEncoder:
    """
    Given an image of a face, compute it's 128D embedding.
    """

    def __init__(self) -> None:
        self.weights_path = os.path.join(
            os.path.dirname(__file__), 'facenet', 'weights.zip')
        self.facenet = FaceNet(self.weights_path)
        self.facenet.load_weights()
        self.facenet.model.eval()

    def calculate_embeddings(self, images: List[Image.Image]) -> np.ndarray:
        """
        Calculate face embeddings

        Parameters
        ----------
        images : list of `PIL.Image.Image`s
            Cropped and resized images of faces.

        Returns
        -------
        embeddings : np.ndarray
            Each list contains 128D representation of a face
        """
        images_np = np.stack([np.array(x) for x in images])
        images_np = np.transpose((images_np - 127.5) / 128.0, (0, 3, 1, 2))

        images_tensors = torch.as_tensor(images_np, dtype=torch.float)
        with torch.no_grad():
            return self.facenet(images_tensors).detach().cpu().numpy()
