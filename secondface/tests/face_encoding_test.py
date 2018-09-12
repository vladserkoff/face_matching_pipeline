#pylint: disable = C0111, R0903, R0201
"""
Tests that all faces are detected, and processed as expected
"""
import os

import numpy as np
from PIL import Image

from secondface.face_encoding import FaceEncoder

TEST_DIR_PATH = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), 'data')
TEST_IMGS = [
    os.path.join(TEST_DIR_PATH, x)
    for x in ['face0.jpg', 'face1.jpg', 'face2.jpg']
]


class Test:
    def test_encoding(self):
        encoder = FaceEncoder()
        images = [Image.open(x) for x in TEST_IMGS]
        embeddings = encoder.calculate_embeddings(images)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 512)
        assert np.allclose(np.linalg.norm(embeddings, axis=1), 1)
